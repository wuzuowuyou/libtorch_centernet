#include <torch/script.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include<cuda_runtime.h>

struct OPT
{
    int input_h;
    int input_w;
    int down_ratio;
    float conf_thresh;
    int k;
};

std::vector<std::string> label_map_big = {
        "hongzhang", "xingshizheng-fuye", "xingshizheng", "chejiahao", "shenfenzheng",
        "shenfenzheng-back", "xingshizheng-back", "cheliangzhaopian", "erweima", "WanShuiZhengMing", "anquandai",
        "fangxiangpan","luntai", "chepailuosi", "mingpai", "mhq", "shanghaianquandai", "xcjly", "yjc",
        "ylb", "fadan", "jiashizheng", "socket", "jiashizheng-fuye"
};

cv::Point2f get_3rd_point(const cv::Point2f &a, const cv::Point2f &b)
{
    cv::Point2f direct = a - b;
    return b + cv::Point2f(-direct.y,direct.x);
}

cv::Mat get_affine_transform(cv::Point2f &center, float &s, cv::Size &output_size, int inv = 0)
{
    int dst_w = output_size.width;
    int dst_h = output_size.height;
    cv::Point2f src_dir = cv::Point2f(0, s * (-0.5));
    cv::Point2f dst_dir = cv::Point2f(0,dst_w*(-0.5));

    cv::Point2f src[3],dst[3];
    src[0] = center;
    src[1] = center + src_dir;
    dst[0] = cv::Point2f(dst_w*1.0*0.5,dst_h*1.0*0.5);
    dst[1] = cv::Point2f(dst_w*1.0*0.5,dst_h*1.0*0.5) + dst_dir;
    src[2] = get_3rd_point(src[0],src[1]);
    dst[2] = get_3rd_point(dst[0],dst[1]);
    cv::Mat trans;
    if(inv)
    {
        trans = cv::getAffineTransform(dst,src);
    }else
    {
        trans = cv::getAffineTransform(src,dst);
    }
    return trans;
}

torch::Tensor pre_process(cv::Mat &img,const OPT &opt)
{
    int height = img.rows;
    int width = img.cols;
    cv::Point2f c = cv::Point2f(width*1.0/2.0,height*1.0/2.0);
    float s = MAX(height*1.0,width*1.0);
    cv::Size out_size = cv::Size(opt.input_w,opt.input_h);
    cv::Mat trans_input = get_affine_transform(c, s, out_size);
//    std::cout<<trans_input<<std::endl;
    cv::Mat inp_image;
    cv::warpAffine(img, inp_image, trans_input, out_size,cv::INTER_LINEAR);

    torch::Tensor tensor_image = torch::from_blob(inp_image.data, { 1, inp_image.rows, inp_image.cols, 3}, torch::kByte).cuda();;
    tensor_image = tensor_image.permute({0,3,1,2});
    tensor_image = tensor_image.toType(torch::kFloat);
    tensor_image = tensor_image.div(255);

    tensor_image[0][0] = tensor_image[0][0].sub_(0.408).div_(0.289);
    tensor_image[0][1] = tensor_image[0][1].sub_(0.447).div_(0.274);
    tensor_image[0][2] = tensor_image[0][2].sub_(0.470).div_(0.278);
//    std::cout<<tensor_image[0][0][0][0]<<std::endl;
//    std::cout<<tensor_image[0][0][256][256]<<std::endl;
    return tensor_image;
}

torch::Tensor nms(const torch::Tensor &heat_tm, int kernel = 3)
{
    int pad = (kernel -1)/2;
    torch::Tensor hmax = torch::max_pool2d(heat_tm, {kernel, kernel}, {1,1}, {pad, pad});
    torch::Tensor keep = (hmax == heat_tm).toType(torch::kFloat32);
    torch::Tensor heat2= heat_tm*keep;
    return heat_tm*keep;
}

torch::Tensor gather_feat(torch::Tensor feat, torch::Tensor ind)
{
    int dim = feat.size(2);
    ind = ind.unsqueeze(2).expand({ind.size(0), ind.size(1), dim});
    feat = feat.gather(1, ind);
    return feat;
}

void _topk(torch::Tensor &scores, torch::Tensor &top_score, torch::Tensor &top_inds, torch::Tensor &top_cls, torch::Tensor &top_ys, torch::Tensor &top_xs, int K=20)
{
    int batch = scores.sizes()[0];
    int cat = scores.sizes()[1];
    int height = scores.sizes()[2];
    int width = scores.sizes()[3];

    std::tuple<torch::Tensor, torch::Tensor> topk_score_inds= topk(scores.view({batch, cat, -1}), K);
    torch::Tensor top_scores = std::get<0>(topk_score_inds);
    top_inds = std::get<1>(topk_score_inds);

    top_inds = top_inds % (height*width);
    top_ys = (top_inds / width).toType(torch::kInt32).toType(torch::kFloat32);
    top_xs = (top_inds % width).toType(torch::kInt32).toType(torch::kFloat32);

    std::tuple<torch::Tensor, torch::Tensor> topk_score_ind = topk(top_scores.view({batch, -1}), K);

    top_score = std::get<0>(topk_score_ind);
    torch::Tensor top_ind = std::get<1>(topk_score_ind);

    top_cls = (top_ind / K).toType(torch::kInt32);
    top_inds = gather_feat(top_inds.view({batch, -1, 1}), top_ind).view({batch, K});
    top_ys = gather_feat(top_ys.view({batch, -1, 1}), top_ind).view({batch, K});
    top_xs = gather_feat(top_xs.view({batch, -1, 1}), top_ind).view({batch, K});
}

torch::Tensor transpose_and_gather(torch::Tensor feat, torch::Tensor ind)
{
    feat = feat.permute({0, 2, 3, 1}).contiguous();
    feat = feat.view({feat.size(0), -1, feat.size(3)});
    feat = gather_feat(feat, ind);
    return feat;
}

void affine_transform(const float &x, const float &y,const cv::Mat &trans, float &x_out, float &y_out)
{
    cv::Mat_<float> mat_pt(3,1);
//    cv::Mat mat_pt(3,1,CV_32F);
    mat_pt(0,0) = x;
    mat_pt(0,1) = y;
    mat_pt(0,2) = 1;
//    std::cout<<trans.type()<<std::endl;
//    std::cout<<mat_pt.type()<<std::endl;
    cv::Mat out = trans * mat_pt;
    x_out = out.at<float>(0,0);
    y_out = out.at<float>(1,0);
}

torch::Tensor ctdet_decode(torch::Tensor &heat, torch::Tensor &wh, torch::Tensor &reg, bool cat_spec_wh=false, int K=100)
{
    heat = nms(heat);
    torch::Tensor scores, inds, cls, ys, xs;
    _topk(heat, scores, inds, cls, ys, xs,K);

    int batch = 1;
    reg = transpose_and_gather(reg,inds);
    reg = reg.view({batch, K, 2});

    xs = xs.view({batch, K, 1}) + reg.slice(2, 0, 1);
    ys = ys.view({batch, K, 1}) + reg.slice(2, 1, 2);

    wh = transpose_and_gather(wh, inds);
    wh = wh.view({batch, K, 2});

    cls = cls.view({batch, K, 1}).toType(torch::kFloat32);
    scores = scores.view({batch, K, 1});

    std::vector<torch::Tensor> vec_tensor = {
            (xs - wh.slice(2,0,1)/2),
            (ys - wh.slice(2,1,2)/2),
            (xs + wh.slice(2,0,1)/2),
            (ys + wh.slice(2,1,2)/2)};
    torch::Tensor bboxes = torch::cat({vec_tensor},2);
    torch::Tensor detection = torch::cat({bboxes, scores, cls}, 2);
    return detection;
}

void ctdet_post_process_my(torch::Tensor &dets, const OPT &opt,cv::Mat &img)
{
    int T_show = 1;
    int height = img.rows;
    int width = img.cols;
    cv::Point2f c = cv::Point2f(width*1.0/2.0,height*1.0/2.0);
    float s = MAX(height*1.0,width*1.0);
    int h = opt.input_h / opt.down_ratio;
    int w = opt.input_w / opt.down_ratio;
    cv::Size size_ = cv::Size(w, h);
    cv::Mat trans = get_affine_transform(c, s,  size_, 1);
    trans.convertTo(trans, CV_32F);
    dets.squeeze_();
    dets = dets.cpu();

    // x1,y1,x2,y2,score,id
    auto result_data = dets.accessor<float, 2>();
    cv::Mat img_draw = img.clone();
    for(int i=0;i<result_data.size(0);i++)
    {
        float score = result_data[i][4];
        if(score < opt.conf_thresh) { continue;}
        float x1 = result_data[i][0];
        float y1 = result_data[i][1];
        float x2 = result_data[i][2];
        float y2 = result_data[i][3];

        affine_transform(x1, y1, trans, x1, y1);
        affine_transform(x2, y2, trans, x2, y2);
        int id_label = result_data[i][5];

        if(T_show)
        {
            cv::rectangle(img_draw,cv::Point(x1,y1),cv::Point(x2,y2),cv::Scalar(255,0,0),1);
            cv::putText(img_draw,label_map_big[id_label],cv::Point(x1,y2),CV_FONT_HERSHEY_SIMPLEX,2,cv::Scalar(0,0,255));
        }
    }
    if(T_show)
    {
        cv::namedWindow("img_draw",0);
        cv::imshow("img_draw",img_draw);
        cv::waitKey(0);
    }
}

void ctdet_post_process_my_save_txt(torch::Tensor &dets, const OPT &opt,cv::Mat &img,std::string path)
{
    int height = img.rows;
    int width = img.cols;
    cv::Point2f c = cv::Point2f(width*1.0/2.0,height*1.0/2.0);
    float s = MAX(height*1.0,width*1.0);
    int h = opt.input_h / opt.down_ratio;
    int w = opt.input_w / opt.down_ratio;
    cv::Size size_ = cv::Size(w, h);
    cv::Mat trans = get_affine_transform(c, s,  size_, 1);
    trans.convertTo(trans, CV_32F);
    dets.squeeze_();
    dets = dets.cpu();

    // x1,y1,x2,y2,score,id
    auto result_data = dets.accessor<float, 2>();
    cv::Mat img_draw = img.clone();
    std::ofstream outfile(path);
    for(int i=0;i<result_data.size(0);i++)
    {
        float score = result_data[i][4];
        if(score < opt.conf_thresh) { continue;}
        float x1 = result_data[i][0];
        float y1 = result_data[i][1];
        float x2 = result_data[i][2];
        float y2 = result_data[i][3];

        affine_transform(x1, y1, trans, x1, y1);
        affine_transform(x2, y2, trans, x2, y2);
        int id_label = result_data[i][5];

        std::string line = label_map_big[id_label] + (std::string)" " + std::to_string(score) + (std::string)" " +\
        (std::string)std::to_string((int)x1) + (std::string)" " \
        + (std::string)std::to_string((int)y1) + (std::string)" " \
        + (std::string)std::to_string((int)x2) + (std::string)" " \
        + (std::string)std::to_string((int)y2) + (std::string)" ";
        if(i != (result_data.size(0)-1))
        {
            line += (std::string)("\n");
        }
        outfile << line;
    }
    outfile.close();
}

int main()
{
    OPT opt;
    opt.input_h = 512;
    opt.input_w = 512;
    opt.down_ratio = 4;
    opt.conf_thresh = 0.2;
    opt.k = 20;
    int flg_show = 1;
    int flg_save_txt = 0;
    std::string path_save_txt_dir = "/data_1/2020biaozhushuju/2020_detection/big/bk/save_txt/";
    std::string model_file = "/data_2/project_202009/pytorch_project/CenterNet/000000experiment_2020/0_1112/CenterNet-master_objvehicle_small_new_test/centernet-big.pt";

    std::shared_ptr<torch::jit::script::Module> module = torch::jit::load(model_file);
    module->eval();
    std::fstream infile("/data_1/2020biaozhushuju/2020_detection/big/test_data/list.txt");
    std::string path_img;
    int cnt = 0;
    auto t_0 = std::chrono::steady_clock::now();
    while(infile >> path_img)
    {
        std::cout<<++cnt << "::"<<path_img<<std::endl;
        int pos_1 = path_img.find_last_of("/");
        std::string name_ = path_img.substr(pos_1+1,path_img.size()-pos_1);
        std::string new_name_txt = name_.substr(0,name_.size()-4) + (std::string)".txt";

        cv::Mat img = cv::imread(path_img);
        torch::Tensor input = pre_process(img,opt);
        auto out = module->forward({input});

        auto tpl = out.toTuple();
        auto out_hm = tpl->elements()[0].toTensor();
//        out_hm.print();
        auto out_wh = tpl->elements()[1].toTensor();
//        out_wh.print();
        auto out_reg = tpl->elements()[2].toTensor();
//        out_reg.print();

        out_hm = torch::sigmoid(out_hm);
        torch::Tensor dets = ctdet_decode(out_hm, out_wh, out_reg,false, opt.k);
        if(flg_show) {ctdet_post_process_my(dets, opt,img);}
        if(flg_save_txt) {ctdet_post_process_my_save_txt(dets, opt,img,path_save_txt_dir + new_name_txt);}
    }
    cudaDeviceSynchronize();
    auto ttt = std::chrono::duration_cast<std::chrono::milliseconds>
            (std::chrono::steady_clock::now() - t_0).count();
    std::cout << "ave consume time="<<ttt*1.0/cnt <<"ms"<<std::endl;
    return 0;
}