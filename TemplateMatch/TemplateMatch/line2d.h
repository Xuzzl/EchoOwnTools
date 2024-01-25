#pragma once

#include <iostream>
#include "fusion.h"
#include "MIPP/mipp.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <map>
#include <mutex>
#include <atomic>
#include <functional>
#include <algorithm>
#include <qcoreapplication.h>

namespace shape_info
{
	class ShapeInfo
	{
	public:
		int angle_step = 10;
		double scale_step = 0.1;
		int angle_range;
		double scale_range;
		cv::Mat src;
		cv::Mat mask;

		class Info
		{
		public:
			int angle;
			double scale;

			Info(int angle_, double scale_) {
				angle = angle_;
				scale = scale_;
			}

			bool operator<(const Info& info) const {
				if (scale != info.scale)
					return scale < info.scale;
				else
					return angle < info.angle;
			}
		};
		std::vector<Info> infos;

		ShapeInfo(const cv::Mat& src_, cv::Mat& mask_)
		{
			src = src_;
			if (mask_.empty()) {
				mask = cv::Mat(src.size(), CV_8UC1, { 255 });
			}
			else {
				mask = mask_;
			}
		}
		static void save_infos(std::vector<ShapeInfo::Info>& infos, std::string path = "infos.yaml")
		{
			cv::FileStorage fs(path, cv::FileStorage::WRITE);

			fs << "infos"
				<< "[";
			for (int i = 0; i < infos.size(); i++) {
				fs << "{";
				fs << "angle" << infos[i].angle;
				fs << "scale" << infos[i].scale;
				fs << "}";
			}
			fs << "]";
		}
		static std::vector<Info> load_infos(std::string path = "infos.yaml")
		{
			cv::FileStorage fs(path, cv::FileStorage::READ);
			std::vector<Info> infos;

			cv::FileNode infos_fn = fs["infos"];
			cv::FileNodeIterator it = infos_fn.begin(), it_end = infos_fn.end();
			for (int i = 0; it != it_end; ++it, ++i) {
				infos.emplace_back(int((*it)["angle"]), double((*it)["scale"]));
			}
			return infos;
		}
		// 生成每个角度与旋转的模板信息
		void produce_infos()
		{
			infos.clear();
			if (0 == angle_step || 0 == scale_step || scale_range < 0)
			{
				return;
			}

			if (0.0 == scale_range) {
				double scale = 1.0;
				for (int angle = -angle_range; angle <= angle_range; angle += angle_step) {
					infos.emplace_back(angle, scale);
				}
			}
			else {
				for (double scale = -scale_range; scale <= scale_range; scale += scale_step) {

					double scale_ = scale + 1.0;
					for (int angle = -angle_range; angle <= angle_range; angle += angle_step) {
						infos.emplace_back(angle, scale_);
					}
				}
			}
		}
		// 为每个角度的旋转和平移作转换
		static cv::Mat transform(cv::Mat& src, int angle, double scale)
		{
			cv::Size image_size;
			cv::Mat rotate_matrix;
			getRotateTransform(src, angle, scale, image_size, rotate_matrix, true);
			// 仿射变换
			cv::Mat output_image;
			cv::warpAffine(src, output_image, rotate_matrix, image_size);
			return output_image;
		}
		cv::Mat src_of(const Info& info)
		{
			return transform(src, info.angle, info.scale);
		}
		cv::Mat mask_of(const Info& info)
		{
			return (transform(mask, info.angle, info.scale) > 0);
		}

		static void getRotateTransform(cv::Mat inputImage, double rotateDeg, double scale, cv::Size& fixSize, cv::Mat& rotateMatrix, bool changeSize)
		{
			/*cv::Rect boundingImage(0, 0, inputImage.cols, inputImage.rows);
			int x, y, w, h;*/

			// 获取旋转矩阵
			// opencv 的角度正方向是逆时针旋转，因此加个"-"号，外部调用函数就是角度正对应顺时针旋转
			cv::Mat rotateMatrixIn = cv::getRotationMatrix2D(cv::Point2f(inputImage.cols / 2, inputImage.rows / 2), -rotateDeg, scale);

			if (!changeSize) {
				fixSize = cv::Size(inputImage.cols, inputImage.rows);
				rotateMatrix = rotateMatrixIn;
				return;
			}

			// 定义图像四个角点数据
			cv::Mat ptsImageCorner = cv::Mat::zeros(4, 3, CV_64FC1);
			ptsImageCorner.at<double>(0, 2) = 1;
			ptsImageCorner.at<double>(1, 0) = inputImage.cols;
			ptsImageCorner.at<double>(1, 2) = 1;
			ptsImageCorner.at<double>(2, 0) = inputImage.cols;
			ptsImageCorner.at<double>(2, 1) = inputImage.rows;
			ptsImageCorner.at<double>(2, 2) = 1;
			ptsImageCorner.at<double>(3, 1) = inputImage.rows;
			ptsImageCorner.at<double>(3, 2) = 1;

			// 旋转之后的四个角点
			cv::Mat vector_rotated = (rotateMatrixIn * (ptsImageCorner.t())).t();

			//找到最小最大的xy
			int x_index = 0, y_index = 1;
			float
				x_min = vector_rotated.at<double>(0, x_index),
				x_max = vector_rotated.at<double>(0, x_index),
				y_min = vector_rotated.at<double>(0, y_index),
				y_max = vector_rotated.at<double>(0, y_index);

			for (int i = 1; i < 4; i++) {
				if (x_min > vector_rotated.at<double>(i, x_index)) {
					x_min = vector_rotated.at<double>(i, x_index);
				}
				if (x_max < vector_rotated.at<double>(i, x_index)) {
					x_max = vector_rotated.at<double>(i, x_index);
				}
				if (y_min > vector_rotated.at<double>(i, y_index)) {
					y_min = vector_rotated.at<double>(i, y_index);
				}
				if (y_max < vector_rotated.at<double>(i, y_index)) {
					y_max = vector_rotated.at<double>(i, y_index);
				}
			}

			// 更新旋转后的图像尺寸与旋转矩阵
			cv::Size imageSizeModify(x_max - x_min, y_max - y_min);
			cv::Mat rotateMatrixModify(rotateMatrixIn);
			rotateMatrixModify.at<double>(x_index, 2) -= x_min;
			rotateMatrixModify.at<double>(y_index, 2) -= y_min;

			//vector<Point> pts;
			/*Mat mask;
			Mat resize_homo = findHomography(vector.t(), newsize_vector.t(), mask);*/

			/*cout << "finally rotate matrix :\n" << rotateMatrixModify << endl;
			cout << "image size :\n" << inputImage.cols << "," << inputImage.rows << "=>" << imageSizeModify << endl;*/
			fixSize = imageSizeModify;
			rotateMatrix = rotateMatrixModify;

			return;
		}
	};
} // namespace shape_info

namespace line2Dup
{
	struct Feature
	{
		int x;
		int y;
		int label; // 量化方向
		float theta; // 梯度方向

		void read(const cv::FileNode& fn)
		{
			cv::FileNodeIterator fni = fn.begin();
			fni >> x >> y >> label;
		}
		void write(cv::FileStorage& fs) const { fs << "[:" << x << y << label << "]"; }

		Feature() : x(0), y(0), label(0) {}
		Feature(int _x, int _y, int _label) : x(_x), y(_y), label(_label) {}
	};

	struct Template
	{
		int width;	// 最大x - 最小x坐标
		int height;	// 最大y - 最小y坐标
		int tl_x;	// 最小x坐标
		int tl_y;	// 最小y坐标
		int pyramid_level;
		int idx = 0;
		std::vector<Feature> features;

		bool operator<(const Template& templ) const { return idx < templ.idx; }

		void read(const cv::FileNode& fn)
		{
			width = fn["width"];
			height = fn["height"];
			tl_x = fn["tl_x"];
			tl_y = fn["tl_y"];
			pyramid_level = fn["pyramid_level"];

			cv::FileNode features_fn = fn["features"];
			features.resize(features_fn.size());
			cv::FileNodeIterator it = features_fn.begin(), it_end = features_fn.end();
			for (int i = 0; it != it_end; ++it, ++i) {
				features[i].read(*it);
			}
		}
		void write(cv::FileStorage& fs) const
		{
			fs << "width" << width;
			fs << "height" << height;
			fs << "tl_x" << tl_x;
			fs << "tl_y" << tl_y;
			fs << "pyramid_level" << pyramid_level;
			fs << "features"
				<< "[";
			for (int i = 0; i < (int)features.size(); ++i) {
				features[i].write(fs);
			}
			fs << "]";
		}
	};

	class ColorGradientPyramid
	{
	public:
		size_t num_features;
		int pyramid_level;
		int extract_kernel;
		float weak_threshold;	// 只有梯度幅值高于此值的平方的点才会被量化
		float strong_threshold;	// 候选点梯度幅值必须大于此值的平方

		cv::Mat src;
		cv::Mat mask;
		cv::Mat angle;			// 0-128的量化方向图
		cv::Mat magnitude;		// sobel梯度图
		cv::Mat angle_ori;		// 梯度方向图

		ColorGradientPyramid(const cv::Mat& src, const cv::Mat& mask, float weak_threshold, size_t num_features, float strong_threshold, int extract_kernel);

		// 带有分数的候选特征点
		struct Candidate
		{
			float score;
			Feature f;

			bool operator<(const Candidate& rhs) const { return score > rhs.score; }

			Candidate(int x, int y, int label, float _score) : f(x, y, label), score(_score) {}
		};
		// 返回当前金字塔层对应的量化梯度方向图(angle)供梯度扩散使用
		void quantize(cv::Mat& dst) const;
		// 金字塔下采样处理
		void pyrDown();
		// 量化梯度方向图与幅值图的计算
		void update();
		// 生成当前金字塔层对应的模板
		bool extractTemplate(Template& templ) const;
		// 从候选特征点中筛选相互距离足够的特征点
		static bool selectScatteredFeatures(const std::vector<Candidate>& candidates, std::vector<Feature>& features, size_t num_features, float distance);
	};

	class ColorGradient
	{
	public:
		size_t num_features;
		int extract_kernel;
		float weak_threshold;
		float strong_threshold;

		ColorGradient() {}
		ColorGradient(float _weak_threshold, size_t _num_features, float _strong_threshold, int _extract_kernel)
			: weak_threshold(_weak_threshold), num_features(_num_features), strong_threshold(_strong_threshold), extract_kernel(_extract_kernel) {}

		cv::Ptr<ColorGradientPyramid> process(const cv::Mat& src, const cv::Mat& mask = cv::Mat()) const
		{
			return cv::makePtr<ColorGradientPyramid>(src, mask, weak_threshold, num_features, strong_threshold, extract_kernel);
		}

		void read(const cv::FileNode& fn)
		{
			weak_threshold = fn["weak_threshold"];
			num_features = int(fn["num_features"]);
			strong_threshold = fn["strong_threshold"];
		}
		void write(cv::FileStorage& fs) const
		{
			fs << "weak_threshold" << weak_threshold;
			fs << "num_features" << int(num_features);
			fs << "strong_threshold" << strong_threshold;
		}
	};

	// 主要在fusion中使用
	struct FilterNode
	{
		std::vector<cv::Mat> buffers;
		int num_buf = 1;
		int buffer_rows = 0;
		int buffer_cols = 0;
		int padded_rows = 0;
		int padded_cols = 0;

		// 锚点：其中topleft为完整img
		int anchor_row = 0;  // anchor: where topleft is in full img
		int anchor_col = 0;

		int prepared_row = 0; // where have been calculated in full img
		int prepared_col = 0;
		int parent = -1;

		std::string op_name;
		int op_type = CV_16U;
		int op_r, op_c;

		int simd_step = mipp::N<int16_t>();
		// 使用SIMD的开关
		bool use_simd = true;

		template <class T>
		T* ptr(int r, int c, int buf_idx = 0)
		{
			r -= anchor_row;  // from full img to buffer img
			c -= anchor_col;
			return &buffers[buf_idx].at<T>(r, c);
		}

		// update start_r end_r start_c end_c
		std::function<int(int, int, int, int)> simple_update;
		std::function<int(int, int, int, int)> simd_update;

		// calculate paddings 计算填充(?)
		void backward_rc(std::vector<FilterNode>& nodes, int rows, int cols, int cur_padded_rows, int cur_padded_cols)
		{
			if (rows > buffer_rows) {
				buffer_rows = rows;
				padded_rows = cur_padded_rows;
			}
			if (cols > buffer_cols) {
				buffer_cols = cols;
				padded_cols = cur_padded_cols;
			}
			if (parent >= 0)
				nodes[parent].backward_rc(nodes, buffer_rows + op_r - 1, cols + op_c - 1, cur_padded_rows + op_r / 2, cur_padded_cols + op_c / 2);
		}
	}; // struct FilterNode

	struct Match
	{
		int x;
		int y;
		int template_id;
		float similarity;
		std::string class_id;

		// 重载< 排序匹配, 将相似度高的排在前面
		bool operator<(const Match& rhs) const {
			if (similarity != rhs.similarity)
				return similarity > rhs.similarity;
			else
				return template_id < rhs.template_id;
		}
		bool operator==(const Match& rhs) const { return x == rhs.x && y == rhs.y && similarity == rhs.similarity && class_id == rhs.class_id; }

		Match() {}
		Match(int x, int y, float similarity, const std::string& class_id, int template_id) : x(x), y(y), similarity(similarity), class_id(class_id), template_id(template_id) {}
	};

	class Detector
	{
	public:
		Detector() = default;
		~Detector() = default;

		int numTemplates() const
		{
			int ret = 0;
			TemplatesMap::const_iterator i = class_templates.begin(), iend = class_templates.end();
			for (; i != iend; ++i)
				ret += static_cast<int>(i->second.size());
			return ret;
		}
		int numTemplates(const std::string& class_id) const
		{
			TemplatesMap::const_iterator i = class_templates.find(class_id);
			if (i == class_templates.end())
				return 0;
			return static_cast<int>(i->second.size());
		}
		int numClasses() const
		{
			return static_cast<int>(class_templates.size());
		}
		std::vector<std::string> classIds() const
		{
			std::vector<std::string> ids;
			TemplatesMap::const_iterator i = class_templates.begin(), iend = class_templates.end();
			for (; i != iend; ++i) {
				ids.push_back(i->first);
			}
			return ids;
		}
		void writeClasses(const std::string& format) const
		{
			TemplatesMap::const_iterator it = class_templates.begin(), it_end = class_templates.end();
			for (; it != it_end; ++it) {
				const cv::String& class_id = it->first;
				cv::String filename = cv::format(format.c_str(), class_id.c_str());
				cv::FileStorage fs(filename, cv::FileStorage::WRITE);
				writeClass(class_id, fs);
			}
		}

		void write(cv::FileStorage& fs) const
		{
			fs << "pyramid_levels" << pyramid_levels;
			fs << "T" << T_at_level;
			modality->write(fs);
		}
		void writeClass(const std::string& class_id, cv::FileStorage& fs) const
		{
			TemplatesMap::const_iterator it = class_templates.find(class_id);
			if (it != class_templates.end())
			{
				const std::vector<TemplatePyramid>& tps = it->second;

				fs << "class_id" << it->first;
				fs << "pyramid_levels" << pyramid_levels;
				fs << "template_pyramids" << "[";
				for (size_t i = 0; i < tps.size(); ++i)
				{
					const TemplatePyramid& tp = tps[i];
					fs << "{";
					fs << "template_id" << int(i); //TODO is this cast correct? won't be good if rolls over...
					fs << "templates" << "[";
					for (size_t j = 0; j < tp.size(); ++j)
					{
						fs << "{";
						tp[j].write(fs);
						fs << "}";
					}
					fs << "]";
					fs << "}";
				}
				fs << "]";
			}
		}

		void read(const cv::FileNode& fn);
		std::string readClass(const cv::FileNode& fn, const std::string& class_id);
		void readClasses(const std::string& class_id, const std::string& format);

		std::vector<Template> getTemplates(const std::string& class_id, int template_id) const;

		// 获取每一次旋转与缩放后的图片的特征点集和坐标，存入模板中
		int addTemplate(shape_info::ShapeInfo& shapes, const std::string& class_id, std::vector<shape_info::ShapeInfo::Info>& infos);

		struct TemplateParams
		{
			int num_features = 128;
			int T = 2;
			float weak_thresh = 30;
			float strong_thresh = 60;
			int angle_range = 10;
			int angle_step = 10;
			double scale_range = 0.0;
			double scale_step = 0.1;
			int extract_kernel = 3;
		};

		struct TemplateInfos
		{
			std::string template_id;
			std::vector<shape_info::ShapeInfo::Info> shape_infos;	// 模板数据（角度和尺度）
			cv::Mat image;
			cv::Mat mask;
			TemplateParams params;
		};
		typedef std::vector<Template> TemplatePyramid;
		typedef std::map<std::string, std::vector<TemplatePyramid>> TemplatesMap;
		typedef std::vector<cv::Mat> LinearMemories;
		// Indexed as [pyramid level][ColorGradient][quantized label]
		typedef std::vector<std::vector<LinearMemories>> LinearMemoryPyramid;

		TemplateParams getParam(std::string id) const;

		// 用于过滤弱匹配
		struct MatchPredicate
		{
			float threshold;
			bool operator()(const Match& m) { return m.similarity < threshold; }
			MatchPredicate(float _threshold) : threshold(_threshold) {}
		};
		std::vector<std::vector<Match>> match(const cv::Mat& sources, float threshold, const std::vector<TemplateInfos>& template_infos, std::vector<bool> status, const cv::Mat masks = cv::Mat());
		void getMatches(const LinearMemoryPyramid& lm_pyramid, const std::vector<cv::Size>& sizes, float threshold, std::vector<Match>& matches, const std::string& class_id, const std::vector<TemplatePyramid>& template_pyramids) const;

	public:
		// fusion 相关参数
		bool set_produce_dxy = false;
		cv::Mat dx_, dy_;
		std::vector<std::vector<char>> fusion_buffers;

	public:
		bool trainTemplate(const cv::Mat& src, const std::string& id, const std::string& save_dirpath, TemplateParams params);

		//std::vector<MatchResults> matchTemplate(const cv::Mat& src, float match_threshold, std::vector<bool> status, cv::Mat& mat_out, std::vector<std::vector<cv::Point>>& vecpoints);

		bool deleteModelData(const std::string& id);

		bool deleteAllModelData();

		bool setTatlevel(int level);

	public:
		TemplatesMap class_templates;		// 模板数据（特征点x、y、feature 等数据）
		std::vector<TemplateInfos> template_infos;

		cv::Ptr<ColorGradient> modality;	// 模板特征 注意兼容多模板
		float res_map_mag_thresh = 60.0f;	// 强阈值
		int extract_kernel = 5;

		std::vector<int> T_at_level;
		int pyramid_levels;

	};

} // namespace line2Dup
