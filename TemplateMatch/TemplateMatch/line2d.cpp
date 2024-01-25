#if _MSC_VER >= 1600
#pragma execution_character_set("utf-8")
#endif

#include "line2d.h"
using namespace std;

namespace line2Dup
{
	static inline int getLabel(int quantized)
	{
		switch (quantized)
		{
		case 1:
			return 0;
		case 2:
			return 1;
		case 4:
			return 2;
		case 8:
			return 3;
		case 16:
			return 4;
		case 32:
			return 5;
		case 64:
			return 6;
		case 128:
			return 7;
		default:
			qInfo() << ("Invalid value of quantized parameter");
			return -1;
		}
	}

	static cv::Rect cropTemplates(std::vector<Template>& templates)
	{
		int min_x = std::numeric_limits<int>::max();
		int min_y = std::numeric_limits<int>::max();
		int max_x = std::numeric_limits<int>::min();
		int max_y = std::numeric_limits<int>::min();

		// 找到所有金字塔层级的特征的 最大、最小 x,y坐标
		for (int i = 0; i < (int)templates.size(); ++i)
		{
			Template& templ = templates[i];

			for (int j = 0; j < (int)templ.features.size(); ++j)
			{
				int x = templ.features[j].x << templ.pyramid_level;
				int y = templ.features[j].y << templ.pyramid_level;
				min_x = std::min(min_x, x);
				min_y = std::min(min_y, y);
				max_x = std::max(max_x, x);
				max_y = std::max(max_y, y);
			}
		}

		// todo 为什么需要min_x，min_y？
		if (min_x % 2 == 1)
			--min_x;
		if (min_y % 2 == 1)
			--min_y;

		// 设置宽度,高度并根据tl_x、tl_y修正特征点坐标
		for (int i = 0; i < (int)templates.size(); ++i)
		{
			Template& templ = templates[i];

			templ.width = (max_x - min_x) >> templ.pyramid_level;
			templ.height = (max_y - min_y) >> templ.pyramid_level;
			templ.tl_x = min_x >> templ.pyramid_level;
			templ.tl_y = min_y >> templ.pyramid_level;

			for (int j = 0; j < (int)templ.features.size(); ++j)
			{
				templ.features[j].x -= templ.tl_x;
				templ.features[j].y -= templ.tl_y;
			}
		}

		cv::Rect rect = cv::Rect(min_x, min_y, max_x - min_x, max_y - min_y);
		return rect;
	}

	/**
	* @brief  滞后阈值梯度量化处理
	* @param  magnitude			sobel梯度图
	* @param  quantized_angle	0-128的量化方向图
	* @param  angle				梯度方向图
	* @param  threshold			阈值
	*/
	void hysteresisGradient(cv::Mat& magnitude, cv::Mat& quantized_angle, cv::Mat& angle, float threshold)
	{
		//  -- 主要是用于对梯度方向进行量化
		// 在该函数内梯度方向被量化到8个方向（先在360度范围内量化为16个方向然后按论文要求将大于180度的方向映射到180度以内），
		// 同时按论文要求只保留那些在3 * 3邻域内主方向投票数大于4且梯度幅值大于threshold的点的梯度方向。
		cv::Mat_<unsigned char> quantized_unfiltered;
		// 利用convertTo函数对角度进行快速量化，到0-16之间，实际的处理方式不是平均分
		angle.convertTo(quantized_unfiltered, CV_8U, 16.0 / 360.0);

		// 将顶部和底部的像素清零
		memset(quantized_unfiltered.ptr(), 0, quantized_unfiltered.cols);
		memset(quantized_unfiltered.ptr(quantized_unfiltered.rows - 1), 0, quantized_unfiltered.cols);

		// 将第一列和最后一列的像素清零
		for (int r = 0; r < quantized_unfiltered.rows; ++r)
		{
			quantized_unfiltered(r, 0) = 0;
			quantized_unfiltered(r, quantized_unfiltered.cols - 1) = 0;
		}

		// 直接与7(00000111)进行与运算，快速完成方向的映射的同时使[348.75, 360)的角度被量化到0，保证了稳定性
		// 0->0 1->1 ...7->7 8->0 9->1 ...15->7 16->0   注：348.75/360*16=15.5，故[348.75, 360)在convertTo后的值为16(00010000)
		for (int r = 1; r < angle.rows - 1; ++r)
		{
			uchar* quant_r = quantized_unfiltered.ptr<uchar>(r);
			for (int c = 1; c < angle.cols - 1; ++c)
			{
				// 把0-16量化到0-7
				quant_r[c] &= 7;
			}
		}

		// 对原始量化图像进行3 * 3邻域滤波。只接受幅度高于某个阈值的像素，并且在量化上确保局部一致性。
		quantized_angle = cv::Mat::zeros(angle.size(), CV_8U);
		auto range = cv::Range(1, angle.rows - 1);
		for (auto r = range.start; r < range.end; ++r)
		{
			float* mag_r = magnitude.ptr<float>(r);
			for (int c = 1; c < angle.cols - 1; ++c)
			{
				if (mag_r[c] > threshold)
				{
					// 计算像素周围3 * 3区域中的量化直方图
					int histogram[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };

					uchar* patch3x3_row = &quantized_unfiltered(r - 1, c - 1);
					histogram[patch3x3_row[0]]++;
					histogram[patch3x3_row[1]]++;
					histogram[patch3x3_row[2]]++;

					patch3x3_row += quantized_unfiltered.step1();
					histogram[patch3x3_row[0]]++;
					histogram[patch3x3_row[1]]++;
					histogram[patch3x3_row[2]]++;

					patch3x3_row += quantized_unfiltered.step1();
					histogram[patch3x3_row[0]]++;
					histogram[patch3x3_row[1]]++;
					histogram[patch3x3_row[2]]++;

					// 从区域中查找投票最多的方向
					int max_votes = 0;
					int index = -1;
					for (int i = 0; i < 8; ++i)
					{
						if (max_votes < histogram[i])
						{
							index = i;
							max_votes = histogram[i];
						}
					}
					// 只有当区域中的大多数像素方向一致时，才接受进一步量化 (方向占优势的点，进一步量化)
					static const int NEIGHBOR_THRESHOLD = 5;
					if (max_votes >= NEIGHBOR_THRESHOLD)
					{
						// index是方向标识，1<<7就是对应128 得到0-128的quantized_angle图
						*quantized_angle.ptr<uchar>(r, c) = uchar(1 << index);
					}
				}
			}
		}
	}

	/**
	* @brief  quantizedOrientations  量化梯度方向图与幅值图的计算
	* @param  src			输入图片
	* @param  magnitude		sobel梯度图
	* @param  angle			0-128的量化方向图
	* @param  angle_ori		梯度方向图
	* @param  threshold		阈值
	*/
	static void quantizedOrientations(const cv::Mat& src, cv::Mat& magnitude, cv::Mat& angle, cv::Mat& angle_ori, float threshold)
	{
		// 计算得到真实梯度幅值图和方向图后，将其交hysteresisGradient()这个函数进行量化处理 
		cv::Mat smoothed;
		const int KERNEL_SIZE = 7;
		GaussianBlur(src, smoothed, cv::Size(KERNEL_SIZE, KERNEL_SIZE), 0, 0, cv::BORDER_REPLICATE);

		// 分别计算所有颜色通道上的水平和垂直图像导数
		if (src.channels() == 1)
		{
			cv::Mat sobel_dx, sobel_dy;
			cv::Sobel(smoothed, sobel_dx, CV_32F, 1, 0, 3, 1.0, 0.0, cv::BORDER_REPLICATE);  // --------------------- 优化参考 https://www.cnblogs.com/Imageshop/p/7285564.html
			cv::Sobel(smoothed, sobel_dy, CV_32F, 0, 1, 3, 1.0, 0.0, cv::BORDER_REPLICATE);
			magnitude = sobel_dx.mul(sobel_dx) + sobel_dy.mul(sobel_dy);
			cv::phase(sobel_dx, sobel_dy, angle_ori, true);
			hysteresisGradient(magnitude, angle, angle_ori, threshold * threshold);
		}
		else
		{
			magnitude.create(src.size(), CV_32F);

			// Allocate temporary buffers
			cv::Size size = src.size();
			cv::Mat sobel_3dx;              // per-channel horizontal derivative
			cv::Mat sobel_3dy;              // per-channel vertical derivative
			cv::Mat sobel_dx(size, CV_32F); // maximum horizontal derivative
			cv::Mat sobel_dy(size, CV_32F); // maximum vertical derivative
			cv::Mat sobel_ag;               // final gradient orientation (unquantized)

			cv::Sobel(smoothed, sobel_3dx, CV_16S, 1, 0, 3, 1.0, 0.0, cv::BORDER_REPLICATE);
			cv::Sobel(smoothed, sobel_3dy, CV_16S, 0, 1, 3, 1.0, 0.0, cv::BORDER_REPLICATE);

			short* ptrx = (short*)sobel_3dx.data;
			short* ptry = (short*)sobel_3dy.data;
			float* ptr0x = (float*)sobel_dx.data;
			float* ptr0y = (float*)sobel_dy.data;
			float* ptrmg = (float*)magnitude.data;

			const int length1 = static_cast<const int>(sobel_3dx.step1());
			const int length2 = static_cast<const int>(sobel_3dy.step1());
			const int length3 = static_cast<const int>(sobel_dx.step1());
			const int length4 = static_cast<const int>(sobel_dy.step1());
			const int length5 = static_cast<const int>(magnitude.step1());
			const int length0 = sobel_3dy.cols * 3;

			for (int r = 0; r < sobel_3dy.rows; ++r)
			{
				int ind = 0;

				for (int i = 0; i < length0; i += 3)
				{
					// Use the gradient orientation of the channel whose magnitude is largest
					int mag1 = ptrx[i + 0] * ptrx[i + 0] + ptry[i + 0] * ptry[i + 0];
					int mag2 = ptrx[i + 1] * ptrx[i + 1] + ptry[i + 1] * ptry[i + 1];
					int mag3 = ptrx[i + 2] * ptrx[i + 2] + ptry[i + 2] * ptry[i + 2];

					if (mag1 >= mag2 && mag1 >= mag3)
					{
						ptr0x[ind] = ptrx[i];
						ptr0y[ind] = ptry[i];
						ptrmg[ind] = (float)mag1;
					}
					else if (mag2 >= mag1 && mag2 >= mag3)
					{
						ptr0x[ind] = ptrx[i + 1];
						ptr0y[ind] = ptry[i + 1];
						ptrmg[ind] = (float)mag2;
					}
					else
					{
						ptr0x[ind] = ptrx[i + 2];
						ptr0y[ind] = ptry[i + 2];
						ptrmg[ind] = (float)mag3;
					}
					++ind;
				}
				ptrx += length1;
				ptry += length2;
				ptr0x += length3;
				ptr0y += length4;
				ptrmg += length5;
			}

			// Calculate the final gradient orientations
			cv::phase(sobel_dx, sobel_dy, sobel_ag, true);
			hysteresisGradient(magnitude, angle, sobel_ag, threshold * threshold);
			angle_ori = sobel_ag;
		}
	}

	ColorGradientPyramid::ColorGradientPyramid(const cv::Mat& _src, const cv::Mat& _mask, float _weak_threshold, size_t _num_features, float _strong_threshold, int _extract_kernel)
		: src(_src), mask(_mask), pyramid_level(0), weak_threshold(_weak_threshold), num_features(_num_features), strong_threshold(_strong_threshold), extract_kernel(_extract_kernel)
	{
		update();
	}

	void ColorGradientPyramid::quantize(cv::Mat& dst) const
	{
		dst = cv::Mat::zeros(angle.size(), CV_8U);
		angle.copyTo(dst, mask);
	}

	void ColorGradientPyramid::update()
	{
		quantizedOrientations(src, magnitude, angle, angle_ori, weak_threshold);
	}

	void ColorGradientPyramid::pyrDown()
	{
		++pyramid_level;
		num_features /= 2;
		cv::Size size(src.cols / 2, src.rows / 2);
		cv::Mat next_src;
		cv::pyrDown(src, next_src, size);
		src = next_src;

		if (!mask.empty())
		{
			cv::Mat next_mask;
			cv::resize(mask, next_mask, size, 0.0, 0.0, cv::INTER_NEAREST);
			mask = next_mask;
		}
		update();
	}

	bool ColorGradientPyramid::extractTemplate(Template& templ) const
	{
		// 将边界上的特征与背景区分
		cv::Mat local_mask;
		if (!mask.empty())
		{
			int erosion_size = 3;    // change object mask into search mask: lcq 
			cv::Mat element = getStructuringElement(cv::MORPH_RECT, cv::Size(2 * erosion_size + 1, 2 * erosion_size + 1), cv::Point(erosion_size, erosion_size));
			cv::erode(mask, local_mask, element, cv::Point(-1, -1), 1, cv::BORDER_REPLICATE);
		}

		std::vector<Candidate> candidates;
		bool no_mask = local_mask.empty();
		float threshold_sq = strong_threshold * strong_threshold;
		cv::Mat magnitude_valid = cv::Mat(magnitude.size(), CV_8UC1, cv::Scalar(255));

		// 遍历 magnitude 的每个像素，如果当前像素数值大于0，且其5*5邻域内有像素的梯度幅值超过它，那么is_max为false, 
		// 如果遍历完后，is_max为true, 那么所有邻域像素对应的magnitude_valid的值，置为0
		for (int r = 0 + extract_kernel / 2; r < magnitude.rows - extract_kernel / 2; ++r)
		{
			const uchar* angle_r = angle.ptr<uchar>(r);
			const uchar* mask_r = no_mask ? NULL : local_mask.ptr<uchar>(r);
			for (int c = 0 + extract_kernel / 2; c < magnitude.cols - extract_kernel / 2; ++c)
			{
				if (no_mask || mask_r[c])
				{
					uchar quantized = angle_r[c];
					float score = 0;
					if (*magnitude_valid.ptr<uchar>(r, c) > 0)
					{
						score = *magnitude.ptr<float>(r, c);
						bool is_max = true;
						for (int r_offset = -extract_kernel / 2; r_offset <= extract_kernel / 2; r_offset++)
						{
							for (int c_offset = -extract_kernel / 2; c_offset <= extract_kernel / 2; c_offset++)
							{
								if (r_offset == 0 && c_offset == 0)
									continue;
								if (score < *magnitude.ptr<float>(r + r_offset, c + c_offset))
								{
									score /= extract_kernel;
									is_max = false;
									break;
								}
							}
							if (!is_max)
								break;
						}
						if (is_max) {
							for (int r_offset = -extract_kernel / 2; r_offset <= extract_kernel / 2; r_offset++)
							{
								for (int c_offset = -extract_kernel / 2; c_offset <= extract_kernel / 2; c_offset++)
								{
									if (r_offset == 0 && c_offset == 0)
										continue;
									*magnitude_valid.ptr<uchar>(r + r_offset, c + c_offset) = 0;
								}
							}
						}
					}
					// 如果幅值超过阈值，且方向不为0，存入
					if (score > threshold_sq && quantized > 0)
					{
						candidates.emplace_back(Candidate(c, r, getLabel(quantized), score));
						candidates.back().f.theta = *angle_ori.ptr<float>(r, c);
					}
				}
			}
		}

		if (candidates.size() < num_features)
		{
			if (candidates.size() <= 4)
			{
				qInfo() << "too few features, abort";
				return false;
			}
			qInfo() << "have no enough features, exaustive mode";
		}
		std::stable_sort(candidates.begin(), candidates.end());

		float distance = static_cast<float>(candidates.size() / num_features + 1);
		selectScatteredFeatures(candidates, templ.features, num_features, distance);

		// 尺寸由外部确定，需要匹配其他模式的模板
		templ.width = -1;
		templ.height = -1;
		templ.pyramid_level = pyramid_level;
		return true;
	}

	bool ColorGradientPyramid::selectScatteredFeatures(const std::vector<Candidate>& candidates, std::vector<Feature>& features, size_t num_features, float distance)
	{
		// 特征点的选择采用一种启发式的方式：首先取score最高的候选点作为第一个特征点，循环选择与已选特征点满足距离限制
		// 并且score最高的点。当选择完毕时特征点的数量小于num_features时，distance会被缩小以放松选择条件保证选取足够的点。
		features.clear();
		float distance_sq = distance;// *distance;
		int i = 0;
		bool first_select = true;

		while (true)
		{
			Candidate c = candidates[i];
			// 如果与之前选择的任何特征有足够的距离，则添加
			bool keep = true;
			for (int j = 0; (j < (int)features.size()) && keep; ++j)
			{
				Feature f = features[j];
				keep = (c.f.x - f.x) * (c.f.x - f.x) + (c.f.y - f.y) * (c.f.y - f.y) >= distance_sq;
			}
			if (keep)
				features.push_back(c.f);

			if (++i == (int)candidates.size())
			{
				bool num_ok = features.size() >= num_features;
				//if (first_select) {
				//	if (num_ok) {
				//		features.clear(); // 特征点数量超过需要的特征点，则清空，加大距离，再次筛选
				//		i = 0;
				//		distance += 1.0f;
				//		distance_sq = distance * distance;
				//		continue;
				//	}
				//	else {
				//		first_select = false;
				//	}
				//}
				// 最后一次筛选，放宽所需距离
				i = 0;
				distance -= 1.0f;
				distance_sq = distance;// *distance;
				if (num_ok || distance < 3)
				{
					break;
				}
			}
		}
		return true;
	}

#pragma region Response maps
	static void orUnaligned8u(const uchar* src, const int src_stride, uchar* dst, const int dst_stride, const int width, const int height)
	{
		// OpenCV 这里再一次展现了实现技巧， 最直观的方法是 每次遍历一个像素时，取出其所有邻域内的像素的梯度方向值，然后做一个或运算， 这样做 内存访问性能较低，
		// 因为图像的下一行和上一行 距离较大， 很可能缓存命中失败。OpenCV 的做法是： 每次遍历时， 只做整个邻域内某个特定位置的像素梯度方向值 的 或运算，
		// 这个地方说的邻域包含像素自身，即邻域中心。 所以总共循环 T*T次。 T 为邻域直径。 这样做， 内存访问友好，并且方便使用 SSE指令进行优化， 
		// 因为连续参与运算的数据在内存中是连续的！
		for (int r = 0; r < height; ++r)
		{
			int c = 0;

			// not aligned, which will happen because we move 1 bytes a time for spreading
			while (reinterpret_cast<unsigned long long>(src + c) % 16 != 0)
			{
				dst[c] |= src[c];
				c++;
			}

			// avoid out of bound when can't divid
			// note: can't use c<width !!!
			for (; c <= width - mipp::N<uint8_t>(); c += mipp::N<uint8_t>())
			{
				mipp::Reg<uint8_t> src_v((uint8_t*)src + c);
				mipp::Reg<uint8_t> dst_v((uint8_t*)dst + c);
				mipp::Reg<uint8_t> res_v = mipp::orb(src_v, dst_v);
				res_v.store((uint8_t*)dst + c);
			}

			for (; c < width; c++)
				dst[c] |= src[c];

			// 前进到下一行
			src += src_stride;
			dst += dst_stride;
		}
	}

	// 梯度方向展开方向扩散。对被搜索图像（原图）也要进行梯度的计算，并且对梯度的方向做方向的拓展。
	// (即: 把每个像素及其邻域的离散化的梯度方向进行 或运算) ，继而利用模板进行滑窗匹配时就有了一定的容错度，匹配容错。
	static void spread(const cv::Mat& src, cv::Mat& dst, int T)
	{
		dst = cv::Mat::zeros(src.size(), CV_8U);

		// 填充扩散梯度图像（论文第2.3节）
		for (int r = 0; r < T; ++r)
		{
			for (int c = 0; c < T; ++c)
			{
				orUnaligned8u(&src.at<unsigned char>(r, c), static_cast<const int>(src.step1()), dst.ptr(), static_cast<const int>(dst.step1()), src.cols - c, src.rows - r);
			}
		}
	}
	/**
	* 将所有可能出现的方向“string”（比如像00101100这样代表方向的二进制数）和某个特定的方向进行角度差计算，得到一个LUT查找表。得到了表以后，
	* 在计算方向某个特定方向响应的时候就只需要查找对应位置的值就可以了，比如0 - 32对应第一个方向“— > ”（水平方向），0 - 16表示0 - 3角度差，16 - 32表示4 - 7角度差。
	* SSE花了时间了解一下，加速来自于CPU某些特定的寄存器，能够一次取一个很长的地址的值进行运算，比如计算4个浮点数，通过__m128i*指针进行计算就能单条指令计算出结果，
	* 相当于原来4个语句，不过实际加速效果没有4倍，编译器在某些情况下也会自己进行SSE加速。

		预处理响应图
		> 制作查找表，算法匹配速度快；
		> 针对n(=8)个方向和方向扩散图逐个像素进行匹配，匹配的结果是距离最近方向角度的余弦值；
		> 值得注意的是，虽然这里有八个方向，但是夹角只有五种情况（算的直线夹角而非射线），故而匹配的结果只有五种；
		> 响应图是被搜索图（原图）各个位置下对应扩展方向的二进制表示，模板图像共有5个方向，那么相应生成5张响应图Response Maps，
			利用模板进行匹配，可以直接调用对应方向在对应像素的结果，避免了滑窗时重复的计算。
	*/
	static const unsigned char LUT3 = 3;
	// 1,2-->0 3-->LUT3
	CV_DECL_ALIGNED(16)
		static const unsigned char SIMILARITY_LUT[256] = { 0, 4, LUT3, 4, 0, 4, LUT3, 4, 0, 4, LUT3, 4, 0, 4, LUT3, 4, 0, 0, 0, 0, 0, 0, 0, 0, LUT3, LUT3, LUT3, LUT3,
		LUT3, LUT3, LUT3, LUT3, 0, LUT3, 4, 4, LUT3, LUT3, 4, 4, 0, LUT3, 4, 4, LUT3, LUT3, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, LUT3, LUT3, 4,
		4, 4, 4, LUT3, LUT3, LUT3, LUT3, 4, 4, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, LUT3, LUT3, LUT3, LUT3, 4, 4, 4, 4, 4, 4, 4, 4, 0, LUT3,
		0, LUT3, 0, LUT3, 0, LUT3, 0, LUT3, 0, LUT3, 0, LUT3, 0, LUT3, 0, 0, 0, 0, 0, 0, 0, 0, LUT3, LUT3, LUT3, LUT3, LUT3, LUT3, LUT3, LUT3, 0, 4, LUT3, 4, 0, 4, LUT3,
		4, 0, 4, LUT3, 4, 0, 4, LUT3, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, LUT3, 4, 4, LUT3, LUT3, 4, 4, 0, LUT3, 4, 4, LUT3, LUT3, 4, 4, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, LUT3,LUT3, 4, 4, 4, 4, LUT3, LUT3, LUT3, LUT3, 4, 4, 4, 4, 0, LUT3, 0, LUT3, 0, LUT3, 0, LUT3, 0, LUT3, 0, LUT3, 0, LUT3,
		0, LUT3, 0, 0, 0, 0, LUT3, LUT3,LUT3, LUT3, 4, 4, 4, 4, 4, 4, 4, 4 };
	/**
	* computeResponseMaps   梯度响应计算
	* @param  src			Mat					输入的是量化并且扩散的梯度图
	* @param  response_maps	std::vector<Mat>	8张不同方向梯度的响应图，顾名思义就是这些图的值越大，匹配度就越高
	*/
	static void computeResponseMaps(const cv::Mat& src, std::vector<cv::Mat>& response_maps)
	{
		CV_Assert((src.rows * src.cols) % 16 == 0);

		// 分配响应图
		response_maps.resize(8);
		for (int i = 0; i < 8; ++i)
			response_maps[i].create(src.size(), CV_8U);

		cv::Mat lsb4(src.size(), CV_8U);
		cv::Mat msb4(src.size(), CV_8U);

		for (int r = 0; r < src.rows; ++r)
		{
			const uchar* src_r = src.ptr(r);
			uchar* lsb4_r = lsb4.ptr(r);
			uchar* msb4_r = msb4.ptr(r);

			for (int c = 0; c < src.cols; ++c)
			{
				// Least significant 4 bits of spread image pixel
				// 扩展图像像素的最低有效4bit
				lsb4_r[c] = src_r[c] & 15;
				// Most significant 4 bits, right-shifted to be in [0, 16)
				// 最高有效4bit，右移到[0，16）
				msb4_r[c] = (src_r[c] & 240) >> 4;
			}
		}

		uchar* lsb4_data = lsb4.ptr<uchar>();
		uchar* msb4_data = msb4.ptr<uchar>();

		bool no_max = true;
		bool no_shuff = true;

#ifdef has_max_int8_t
		no_max = false;
#endif
#ifdef has_shuff_int8_t
		no_shuff = false;
#endif
		// LUT is designed for 128 bits SIMD, so quite triky for others

		// 遍历8个量化方向中的每一个
		for (int ori = 0; ori < 8; ++ori)
		{
			uchar* map_data = response_maps[ori].ptr<uchar>();
			const uchar* lut_low = SIMILARITY_LUT + 32 * ori;

			if (mipp::N<uint8_t>() == 1 || no_max || no_shuff)
			{
				// no SIMD
				for (int i = 0; i < src.rows * src.cols; ++i)
					map_data[i] = std::max(lut_low[lsb4_data[i]], lut_low[msb4_data[i] + 16]);
			}
			else if (mipp::N<uint8_t>() == 16)
			{
				// 128 SIMD, no add base

				const uchar* lut_low = SIMILARITY_LUT + 32 * ori;
				mipp::Reg<uint8_t> lut_low_v((uint8_t*)lut_low);
				mipp::Reg<uint8_t> lut_high_v((uint8_t*)lut_low + 16);

				for (int i = 0; i < src.rows * src.cols; i += mipp::N<uint8_t>())
				{
					mipp::Reg<uint8_t> low_mask((uint8_t*)lsb4_data + i);
					mipp::Reg<uint8_t> high_mask((uint8_t*)msb4_data + i);

					mipp::Reg<uint8_t> low_res = mipp::shuff(lut_low_v, low_mask);
					mipp::Reg<uint8_t> high_res = mipp::shuff(lut_high_v, high_mask);

					mipp::Reg<uint8_t> result = mipp::max(low_res, high_res);
					result.store((uint8_t*)map_data + i);
				}
			}
			else if (mipp::N<uint8_t>() == 16 || mipp::N<uint8_t>() == 32 || mipp::N<uint8_t>() == 64) { //128 256 512 SIMD
				CV_Assert((src.rows * src.cols) % mipp::N<uint8_t>() == 0);

				uint8_t lut_temp[mipp::N<uint8_t>()] = { 0 };

				for (int slice = 0; slice < mipp::N<uint8_t>() / 16; slice++)
				{
					std::copy_n(lut_low, 16, lut_temp + slice * 16);
				}
				mipp::Reg<uint8_t> lut_low_v(lut_temp);

				uint8_t base_add_array[mipp::N<uint8_t>()] = { 0 };
				for (uint8_t slice = 0; slice < mipp::N<uint8_t>(); slice += 16)
				{
					std::copy_n(lut_low + 16, 16, lut_temp + slice);
					std::fill_n(base_add_array + slice, 16, slice);
				}
				mipp::Reg<uint8_t> base_add(base_add_array);
				mipp::Reg<uint8_t> lut_high_v(lut_temp);

				for (int i = 0; i < src.rows * src.cols; i += mipp::N<uint8_t>())
				{
					mipp::Reg<uint8_t> mask_low_v((uint8_t*)lsb4_data + i);
					mipp::Reg<uint8_t> mask_high_v((uint8_t*)msb4_data + i);

					mask_low_v += base_add;
					mask_high_v += base_add;

					mipp::Reg<uint8_t> shuff_low_result = mipp::shuff(lut_low_v, mask_low_v);
					mipp::Reg<uint8_t> shuff_high_result = mipp::shuff(lut_high_v, mask_high_v);

					mipp::Reg<uint8_t> result = mipp::max(shuff_low_result, shuff_high_result);
					result.store((uint8_t*)map_data + i);
				}
			}
			else
			{
				for (int i = 0; i < src.rows * src.cols; ++i)
					map_data[i] = std::max(lut_low[lsb4_data[i]], lut_low[msb4_data[i] + 16]);
			}
		}
	}

	// 线性化存储  避免重复计算，加速计算  (改变存储方式，先行后列， 间隔T读取，然后写入，没有比较复杂和特殊的处理)
	static void linearize(const cv::Mat& response_map, cv::Mat& linearized, int T)
	{
		CV_Assert(response_map.rows % T == 0);
		CV_Assert(response_map.cols % T == 0);

		// 线性化有 T^2 行，其中每行都是一个线性存储器
		int mem_width = response_map.cols / T;
		int mem_height = response_map.rows / T;
		linearized.create(T * T, mem_width * mem_height, CV_8U);

		// Outer two for loops iterate over top-left T^2 starting pixels
		// 外部两个for循环在左上角 T^2 个起始像素上迭代
		int index = 0;
		for (int r_start = 0; r_start < T; ++r_start)
		{
			for (int c_start = 0; c_start < T; ++c_start)
			{
				uchar* memory = linearized.ptr(index);
				++index;
				// Inner two loops copy every T-th pixel into the linear memory
				// 内部两个循环将每个第T个像素复制到线性存储器中
				for (int r = r_start; r < response_map.rows; r += T)
				{
					const uchar* response_data = response_map.ptr(r);
					for (int c = c_start; c < response_map.cols; c += T)
						*memory++ = response_data[c];
				}
			}
		}
	}
#pragma endregion

#pragma region Linearized similarities
	/**
	* @brief  根据模板中的特征点 获得某个特定方向的 整个线性化的响应图的响应大小
	* @param  linear_memories	线性存储器
	* @param  f					特征
	* @param  T					金字塔系数
	* @param  W					被系数T缩小的输入图像宽度
	*/
	static const unsigned char* accessLinearMemory(const std::vector<cv::Mat>& linear_memories, const Feature& f, int T, int W)
	{
		// 检索与特征标签相关的线性存储器的 T*T 网格
		const cv::Mat& memory_grid = linear_memories[f.label];
		CV_DbgAssert(memory_grid.rows == T * T);
		CV_DbgAssert(f.x >= 0);
		CV_DbgAssert(f.y >= 0);

		// 我们想要的 LinearMemory 位于T*T网格中的（x%T，y%T）（存储为memory_grid的行）
		int grid_x = f.x % T;
		int grid_y = f.y % T;
		int grid_index = grid_y * T + grid_x;
		CV_DbgAssert(grid_index >= 0);
		CV_DbgAssert(grid_index < memory_grid.rows);
		const unsigned char* memory = memory_grid.ptr(grid_index);

		// 在LinearMemory中，特征位于（x/T，y/T）。W是LM的“宽度”，即系数T缩小的输入图像宽度。
		int lm_x = f.x / T;
		int lm_y = f.y / T;
		int lm_index = lm_y * W + lm_x;
		CV_DbgAssert(lm_index >= 0);
		CV_DbgAssert(lm_index < memory_grid.cols);
		return memory + lm_index;
	}

	/**
	* @brief  计算模板和输入图像的 相似性
	* @param  linear_memories	线性存储器
	* @param  templ				模板
	* @param  dst
	* @param  size				输入图像尺寸
	* @param  T					金字塔系数
	*/
	static void similarity(const std::vector<cv::Mat>& linear_memories, const Template& templ, cv::Mat& dst, cv::Size size, int T)
	{
		// -------------------------------代码和 similarity_64 相似，注释看 similarity_64
		// 我们只有一个模态，所以8192*2，由于mipp，回到8192
		if (templ.features.size() > 8192)
		{
			return;
		}

		int W = size.width / T;
		int H = size.height / T;
		int wf = (templ.width - 1) / T + 1;
		int hf = (templ.height - 1) / T + 1;
		int span_x = W - wf;
		int span_y = H - hf;
		int template_positions = span_y * W + span_x + 1;

		dst = cv::Mat::zeros(H, W, CV_16U);
		short* dst_ptr = dst.ptr<short>();
		mipp::Reg<uint8_t> zero_v(uint8_t(0));

		for (int i = 0; i < (int)templ.features.size(); ++i)
		{
			Feature f = templ.features[i];
			if (f.x < 0 || f.x >= size.width || f.y < 0 || f.y >= size.height)
				continue;
			const uchar* lm_ptr = accessLinearMemory(linear_memories, f, T, W);

			int j = 0;
			// *2 避免int8 读取超出范围
			for (; j <= template_positions - mipp::N<int16_t>() * 2; j += mipp::N<int16_t>())
			{
				mipp::Reg<uint8_t> src8_v((uint8_t*)lm_ptr + j);

				// uchar to short, once for N bytes
				mipp::Reg<int16_t> src16_v(mipp::interleavelo(src8_v, zero_v).r);
				mipp::Reg<int16_t> dst_v((int16_t*)dst_ptr + j);

				mipp::Reg<int16_t> res_v = src16_v + dst_v;
				res_v.store((int16_t*)dst_ptr + j);
			}
			for (; j < template_positions; j++)
				dst_ptr[j] += short(lm_ptr[j]);
		}
	}

	static void similarityLocal(const std::vector<cv::Mat>& linear_memories, const Template& templ, cv::Mat& dst, cv::Size size, int T, cv::Point center)
	{
		// -------------------------------代码和 similarityLocal_64 相似，注释看 similarityLocal_64
		if (templ.features.size() > 8192)
		{
			return;
		}

		int W = size.width / T;
		dst = cv::Mat::zeros(16, 16, CV_16U);

		int offset_x = (center.x / T - 8) * T;
		int offset_y = (center.y / T - 8) * T;
		mipp::Reg<uint8_t> zero_v = uint8_t(0);

		for (int i = 0; i < (int)templ.features.size(); ++i)
		{
			Feature f = templ.features[i];
			f.x += offset_x;
			f.y += offset_y;
			// 如果超出边界 放弃特征，可能是由于应用偏移
			if (f.x < 0 || f.y < 0 || f.x >= size.width || f.y >= size.height)
				continue;

			const uchar* lm_ptr = accessLinearMemory(linear_memories, f, T, W);
			short* dst_ptr = dst.ptr<short>();

			if (mipp::N<uint8_t>() > 32) { //512 bits SIMD
				for (int row = 0; row < 16; row += mipp::N<int16_t>() / 16)
				{
					mipp::Reg<int16_t> dst_v((int16_t*)dst_ptr + row * 16);

					// load lm_ptr, 16 bytes once, for half
					uint8_t local_v[mipp::N<uint8_t>()] = { 0 };
					for (int slice = 0; slice < mipp::N<uint8_t>() / 16 / 2; ++slice)
					{
						std::copy_n(lm_ptr, 16, &local_v[16 * slice]);
						lm_ptr += W;
					}
					mipp::Reg<uint8_t> src8_v(local_v);
					// uchar to short, once for N bytes
					mipp::Reg<int16_t> src16_v(mipp::interleavelo(src8_v, zero_v).r);

					mipp::Reg<int16_t> res_v = src16_v + dst_v;
					res_v.store((int16_t*)dst_ptr);

					dst_ptr += mipp::N<int16_t>();
				}
			}
			else
			{ // 256 128 or no SIMD
				for (int row = 0; row < 16; ++row)
				{
					for (int col = 0; col < 16; col += mipp::N<int16_t>())
					{
						mipp::Reg<uint8_t> src8_v((uint8_t*)lm_ptr + col);

						// uchar to short, once for N bytes
						mipp::Reg<int16_t> src16_v(mipp::interleavelo(src8_v, zero_v).r);

						mipp::Reg<int16_t> dst_v((int16_t*)dst_ptr + col);
						mipp::Reg<int16_t> res_v = src16_v + dst_v;
						res_v.store((int16_t*)dst_ptr + col);
					}
					dst_ptr += 16;
					lm_ptr += W;
				}
			}
		}
	}

	/**
	* @brief  计算特征点少于64的模板和输入图像的相似性
	* @param  linear_memories	线性存储器
	* @param  templ				模板
	* @param  dst
	* @param  size				输入图像尺寸
	* @param  T					金字塔系数
	*/
	static void similarity_64(const std::vector<cv::Mat>& linear_memories, const Template& templ, cv::Mat& dst, cv::Size size, int T)
	{
		// 63个或更少的特征是一种特殊情况，因为每个特征的最大相似性是4。255/4=63，所以最多可以将相似性加在8位中，
		// 而不用担心溢出。因此，这里我们使用_mm_add_epi8作为主力，而更通用的函数将使用_mm_add_epi16。
		CV_Assert(templ.features.size() < 64);

		// 按系数T决定输入图像大小
		int W = size.width / T;
		int H = size.height / T;

		// 特征尺寸，按系数T缩小并四舍五入
		int wf = (templ.width - 1) / T + 1;
		int hf = (templ.height - 1) / T + 1;

		// Span是我们可以围绕输入图像移动模板的范围
		int span_x = W - wf;
		int span_y = H - hf;

		// 计算在图像上滑动特征时要检查的连续（内存中）像素数。这允许模板不正确地环绕左/右边界，因此必须过滤掉任何已环绕的模板匹配！
		int template_positions = span_y * W + span_x + 1; // @todo 为啥加1?
		//int template_positions = (span_y - 1) * W + span_x; // More correct?

		// @todo 在旧代码中，dst是大小为 m_U 的缓冲区。可以改成（span_x）*（span_y）这样吗？
		dst = cv::Mat::zeros(H, W, CV_8U);
		uchar* dst_ptr = dst.ptr<uchar>();

		// 通过累积每个特征的贡献来计算该模板的相似性度量
		for (int i = 0; i < (int)templ.features.size(); ++i) {
			// 从模板中特征的位置 计算出的适当偏移处 添加线性内存
			Feature f = templ.features[i];

			// 如果超出范围，则放弃特征
			if (f.x < 0 || f.x >= size.width || f.y < 0 || f.y >= size.height) //@todo 实际上看不到x<0 y＜0 ？
				continue;
			const uchar* lm_ptr = accessLinearMemory(linear_memories, f, T, W);

			// 现在，我们使用template_positions元素对dst_ptr和lm_ptr进行对齐/未对齐的添加
			int j = 0;
			for (; j <= template_positions - mipp::N<uint8_t>(); j += mipp::N<uint8_t>()) {
				mipp::Reg<uint8_t> src_v((uint8_t*)lm_ptr + j);
				mipp::Reg<uint8_t> dst_v((uint8_t*)dst_ptr + j);

				mipp::Reg<uint8_t> res_v = src_v + dst_v;
				res_v.store((uint8_t*)dst_ptr + j);
			}
			for (; j < template_positions; j++)
				dst_ptr[j] += lm_ptr[j];
		}
	}

	static void similarityLocal_64(const std::vector<cv::Mat>& linear_memories, const Template& templ, cv::Mat& dst, cv::Size size, int T, cv::Point center)
	{
		// 类似于上面的similarity() 此版本采用位置“中心”，并计算以其为中心的16*16 patch中的能量 
		CV_Assert(templ.features.size() < 64);

		// 计算中心周围16*16 patch 中的相似性图
		int W = size.width / T;
		dst = cv::Mat::zeros(16, 16, CV_8U);

		// NOTE: We make the offsets multiples of T to agree with results of the original code.
		// 偏移每个特征点所请求的中心位置  从中心调整到（-8，-8），得到左上角的16*16 patch
		// 注意：我们将偏移量设为T的倍数，与原始代码的结果一致 
		int offset_x = (center.x / T - 8) * T;
		int offset_y = (center.y / T - 8) * T;

		for (int i = 0; i < (int)templ.features.size(); ++i) {
			Feature f = templ.features[i];
			f.x += offset_x;
			f.y += offset_y;
			// 如果超出边界 放弃特征，可能是由于应用偏移
			if (f.x < 0 || f.y < 0 || f.x >= size.width || f.y >= size.height)
				continue;

			const uchar* lm_ptr = accessLinearMemory(linear_memories, f, T, W);
			uchar* dst_ptr = dst.ptr<uchar>();

			if (mipp::N<uint8_t>() > 16) { // 256 or 512 bits SIMD
				for (int row = 0; row < 16; row += mipp::N<uint8_t>() / 16) {
					mipp::Reg<uint8_t> dst_v((uint8_t*)dst_ptr);

					// load lm_ptr, 16 bytes once
					uint8_t local_v[mipp::N<uint8_t>()];
					for (int slice = 0; slice < mipp::N<uint8_t>() / 16; slice++) {
						std::copy_n(lm_ptr, 16, &local_v[16 * slice]);
						lm_ptr += W;
					}
					mipp::Reg<uint8_t> src_v(local_v);
					mipp::Reg<uint8_t> res_v = src_v + dst_v;
					res_v.store((uint8_t*)dst_ptr);

					dst_ptr += mipp::N<uint8_t>();
				}
			}
			else { // 128 or no SIMD
				for (int row = 0; row < 16; ++row) {
					for (int col = 0; col < 16; col += mipp::N<uint8_t>()) {
						mipp::Reg<uint8_t> src_v((uint8_t*)lm_ptr + col);
						mipp::Reg<uint8_t> dst_v((uint8_t*)dst_ptr + col);
						mipp::Reg<uint8_t> res_v = src_v + dst_v;
						res_v.store((uint8_t*)dst_ptr + col);
					}
					dst_ptr += 16;
					lm_ptr += W;
				}
			}
		}
	}
#pragma endregion

#pragma region High-level Detector API

	void Detector::read(const cv::FileNode& fn)
	{
		class_templates.clear();
		pyramid_levels = fn["pyramid_levels"];
		fn["T"] >> T_at_level;
		modality = cv::makePtr<ColorGradient>();
	}

	std::string Detector::readClass(const cv::FileNode& fn, const std::string& class_id)
	{
		if (class_id.empty())
		{
			return std::string();
		}

		TemplatesMap::value_type v(class_id, std::vector<TemplatePyramid>());
		std::vector<TemplatePyramid>& tps = v.second;
		int expected_id = 0;

		cv::FileNode tps_fn = fn["template_pyramids"];
		tps.resize(tps_fn.size());
		cv::FileNodeIterator tps_it = tps_fn.begin(), tps_it_end = tps_fn.end();
		for (; tps_it != tps_it_end; ++tps_it, ++expected_id)
		{
			int template_id = (*tps_it)["template_id"];
			CV_Assert(template_id == expected_id);
			cv::FileNode templates_fn = (*tps_it)["templates"];
			tps[template_id].resize(templates_fn.size());

			cv::FileNodeIterator templ_it = templates_fn.begin(), templ_it_end = templates_fn.end();
			int idx = 0;
			for (; templ_it != templ_it_end; ++templ_it)
			{
				tps[template_id][idx++].read(*templ_it);
			}
		}
		class_templates.insert(v);
		return class_id;
	}

	void Detector::readClasses(const std::string& class_id, const std::string& format)
	{
		cv::String filename = cv::format(format.c_str(), class_id.c_str());
		cv::FileStorage fs(filename, cv::FileStorage::READ);
		readClass(fs.root(), class_id);
	}

	std::vector<Template> Detector::getTemplates(const std::string& class_id, int template_id) const
	{
		auto it = class_templates.find(class_id);
		if (it != class_templates.end() && (it->second.size() > size_t(template_id)))
			return it->second[template_id];
		else
			return std::vector<Template>();
	}

	int Detector::addTemplate(shape_info::ShapeInfo& shapes, const std::string& class_id, std::vector<shape_info::ShapeInfo::Info>& infos)
	{
		int siSize = shapes.infos.size();
		if (siSize < 1 || class_id.empty())
		{
			return -1;
		}

		std::vector<TemplatePyramid>& template_pyramids = class_templates[class_id];
		std::mutex _mutex;

		cv::parallel_for_(cv::Range(0, siSize), [&](const cv::Range& range) {
			//cv::Range range = cv::Range(0, siSize); // 调试用
			for (auto i = range.start; i < range.end; ++i)
			{
				auto& firstInfo = shapes.infos[i];
				auto m1 = shapes.src_of(firstInfo);
				auto m2 = shapes.mask_of(firstInfo);

				TemplatePyramid tp;
				tp.resize(pyramid_levels);
				cv::Ptr<ColorGradientPyramid> qp = modality->process(m1, m2);

				for (int j = 0; j < pyramid_levels; ++j)
				{
					if (j > 0)
						qp->pyrDown();
					bool success = qp->extractTemplate(tp[j]);
					tp[j].idx = i;
					if (!success)
					{
						break;
					}
				}
				cropTemplates(tp);

				std::lock_guard<std::mutex> locker(_mutex);
				template_pyramids.push_back(tp);
				infos.push_back(firstInfo);
			}
			});

		std::sort(template_pyramids.begin(), template_pyramids.end());
		std::sort(infos.begin(), infos.end());
		return 0;
	}

	static int gcd(int a, int b)
	{
		if (a == 0)
			return b;
		return gcd(b % a, a);
	}
	static int lcm(int a, int b)
	{
		return (a * b) / gcd(a, b);
	}
	static int least_mul_of_Ts(const std::vector<int>& T_at_level)
	{
		if (T_at_level.size() < 1)
		{
			return -1;
		}
		int cur_res = T_at_level[0];
		for (int i = 1; i < T_at_level.size(); ++i)
		{
			int cur_v = T_at_level[i] << i;
			cur_res = lcm(cur_v, cur_res);
		}
		return cur_res;
	}

	std::vector<std::vector<Match>> Detector::match(const cv::Mat& source, float threshold, const std::vector<TemplateInfos>& template_infos, std::vector<bool> status, const cv::Mat mask)
	{
		std::vector<std::vector<Match>> all_matches;
		if (source.empty() || template_infos.empty())
		{
			return all_matches;
		}

		// 对于每个金字塔级别，为每个ColorGradient预计算线性内存
		LinearMemoryPyramid lm_pyramid(pyramid_levels, std::vector<LinearMemories>(1, LinearMemories(8)));
		std::vector<cv::Size> sizes;

#pragma region build response maps
		//// 使用 source 初始化每个 ColorGradient
		//std::vector<cv::Ptr<ColorGradientPyramid>> quantizers;
		//quantizers.push_back(modality->process(source, mask));
		//
		//for (int level = 0; level < pyramid_levels; ++level)
		//{
		//	int T = T_at_level[level];
		//	std::vector<LinearMemories> &lm_level = lm_pyramid[level];
		//	if (level > 0) 
		//	{
		//		for (int i = 0; i < (int)quantizers.size(); ++i)
		//			quantizers[i]->pyrDown();
		//	}
		//	cv::Mat quantized, spread_quantized;
		//	std::vector<cv::Mat> response_maps;
		//	for (int i = 0; i < (int)quantizers.size(); ++i) 
		//	{
		//		quantizers[i]->quantize(quantized);
		//		spread(quantized, spread_quantized, T);
		//		computeResponseMaps(spread_quantized, response_maps);
		//		LinearMemories &memories = lm_level[i];
		//		for (int j = 0; j < 8; ++j)
		//			linearize(response_maps[j], memories[j], T);
		//	}
		//	sizes.push_back(quantized.size());
		//}
#pragma endregion

#pragma region build fusion response maps
		dx_ = cv::Mat();
		dy_ = cv::Mat();
		// 现在不需要裁剪，我们在内部进行处理， 让高、宽都是 lcm_Ts 的倍数
		const int lcm_Ts = least_mul_of_Ts(T_at_level);
		const int biggest_imgRows = source.rows / lcm_Ts * lcm_Ts;
		const int biggest_imgCols = source.cols / lcm_Ts * lcm_Ts;

		// fusion 使用参数
		const int tileRows = 32;
		const int tileCols = 256;
		const int num_threads_ = 4;
		//const int num_threads_ = std::thread::hardware_concurrency();
		const int32_t mag_thresh_l2 = int32_t(res_map_mag_thresh * res_map_mag_thresh);

		cv::Mat pyrdown_src;
		for (int level = 0; level < T_at_level.size(); ++level)
		{
			const bool need_pyr = level < T_at_level.size() - 1;
			const int imgRows = biggest_imgRows >> level;
			const int imgCols = biggest_imgCols >> level;
			const int cur_T = T_at_level[level];
			if (cur_T % 2 != 0)
			{
				continue;
			}

			// 使用旧的线性函数创建 
			for (int ori = 0; ori < 8; ++ori)
			{
				lm_pyramid[level][0][ori] = cv::Mat(cur_T * cur_T, imgCols / cur_T * imgRows / cur_T, CV_8U);
			}
			sizes.push_back({ imgCols, imgRows });

			cv::Mat src;
			if (level == 0)
				src = source;
			else
				src = pyrdown_src;

			if (need_pyr)
				pyrdown_src = cv::Mat(imgRows / 2, imgCols / 2, CV_8U);

			simple_fusion::ProcessManager manager(fusion_buffers, tileRows, tileCols);
			manager.set_num_threads(num_threads_);

			if (src.channels() == 3)
				manager.get_nodes().push_back(std::make_shared<simple_fusion::BGR2GRAY_8UC3_8U>());
			manager.get_nodes().push_back(std::make_shared<simple_fusion::Gauss1x5Node_8U_32S_4bit_larger>());
			manager.get_nodes().push_back(std::make_shared<simple_fusion::Gauss5x1withPyrdownNode_32S_16S_4bit_smaller>(pyrdown_src, need_pyr));
			manager.get_nodes().push_back(std::make_shared<simple_fusion::Sobel1x3SxxSyxNode_16S_16S>());

			if (set_produce_dxy && level == 0)
			{
				dx_ = cv::Mat(src.size(), CV_16S, cv::Scalar(0));
				dy_ = cv::Mat(src.size(), CV_16S, cv::Scalar(0));
				manager.get_nodes().push_back(std::make_shared<simple_fusion::Sobel3x1SxySyyNodeWithDxy_16S_16S>(dx_, dy_));
			}
			else
			{
				manager.get_nodes().push_back(std::make_shared<simple_fusion::Sobel3x1SxySyyNode_16S_16S>());
			}

			manager.get_nodes().push_back(std::make_shared<simple_fusion::MagPhaseQuant1x1Node_16S_8U>(mag_thresh_l2));
			manager.get_nodes().push_back(std::make_shared<simple_fusion::Hist3x3Node_8U_8U>());
			manager.get_nodes().push_back(std::make_shared<simple_fusion::Spread1xnNode_8U_8U>(cur_T + 1));
			manager.get_nodes().push_back(std::make_shared<simple_fusion::Spreadnx1Node_8U_8U>(cur_T + 1));
			manager.get_nodes().push_back(std::make_shared<simple_fusion::Response1x1Node_8U_8U>());
			manager.get_nodes().push_back(std::make_shared<simple_fusion::LinearizeTxTNode_8U_8U>(cur_T, imgCols, lm_pyramid[level][0]));

			manager.arrange(imgRows, imgCols);

			std::vector<cv::Mat> in_v;
			in_v.push_back(src);
			std::vector<cv::Mat> out_v = lm_pyramid[level][0];
			manager.process(in_v, out_v);
		}
#pragma endregion

		for (int i = 0; i < (int)template_infos.size(); ++i)
		{
			auto bUse = (status.size() > i) ? status[i] : false;
			if (!bUse)
			{
				all_matches.emplace_back(std::vector<Match>());
				continue;
			}

			auto info = template_infos[i];
			std::vector<Match> matches;

			TemplatesMap::const_iterator it = class_templates.find(info.template_id);
			if (it != class_templates.end())
			{
				getMatches(lm_pyramid, sizes, threshold, matches, it->first, it->second);
			}

			std::sort(matches.begin(), matches.end());
			std::vector<Match>::iterator new_end = std::unique(matches.begin(), matches.end());
			matches.erase(new_end, matches.end());
			all_matches.push_back(matches);
		}
		return all_matches;
	}

	void Detector::getMatches(const LinearMemoryPyramid& lm_pyramid, const std::vector<cv::Size>& sizes, float threshold, std::vector<Match>& matches,
		const std::string& class_id, const std::vector<TemplatePyramid>& template_pyramids) const
	{
		int tpSize = template_pyramids.size();
		if (tpSize < 1)
		{
			return;
		}
		std::mutex _mutex;

		try
		{
			cv::parallel_for_(cv::Range(0, tpSize), [&](const cv::Range& range) {
				//cv::Range range = cv::Range(0, tpSize); // 调试用
				for (auto template_id = range.start; template_id < range.end; ++template_id)
				{
					std::vector<Match> candidates;
					const TemplatePyramid& tp = template_pyramids[template_id];
					if (tp.empty())
					{
						break;
					}

					const std::vector<LinearMemories>& lowest_lm = lm_pyramid.back();

					int lowest_start = static_cast<int>(tp.size() - 1);
					int lowest_T = T_at_level.back();
					int num_features = 0;
					cv::Mat similarities;
					const Template& templ = tp[lowest_start];
					num_features += static_cast<int>(templ.features.size());

					if (templ.features.size() < 64)
					{
						similarity_64(lowest_lm[0], templ, similarities, sizes.back(), lowest_T);
						similarities.convertTo(similarities, CV_16U);
					}
					else if (templ.features.size() < 8192)
					{
						similarity(lowest_lm[0], templ, similarities, sizes.back(), lowest_T);
					}
					else
					{
						qInfo() << ("feature size too large");
					}

					for (int r = 0; r < similarities.rows; ++r)
					{
						ushort* row = similarities.ptr<ushort>(r);
						for (int c = 0; c < similarities.cols; ++c)
						{
							int raw_score = row[c];
							float score = (raw_score * 100.f) / (4 * num_features);
							if (score > threshold)
							{
								int offset = lowest_T / 2 + (lowest_T % 2 - 1);
								int x = c * lowest_T + offset;
								int y = r * lowest_T + offset;
								candidates.emplace_back(Match(x, y, score, class_id, static_cast<int>(template_id)));
							}
						}
					}
					if ((int)candidates.size() == 0)
						continue;

					for (int i = pyramid_levels - 2; i >= 0; --i)
					{
						const std::vector<LinearMemories>& lms = lm_pyramid[i];
						int T = T_at_level[i];
						int start = static_cast<int>(i);
						cv::Size size = sizes[i];
						int border = 8 * T;
						int offset = T / 2 + (T % 2 - 1);
						int max_x = size.width - tp[start].width - border;
						int max_y = size.height - tp[start].height - border;

						cv::Mat similarities2;
						for (int m = 0; m < (int)candidates.size(); ++m)
						{
							Match& match2 = candidates[m];
							int x = match2.x * 2 + 1;
							int y = match2.y * 2 + 1;

							x = std::max(x, border);
							y = std::max(y, border);
							x = std::min(x, max_x);
							y = std::min(y, max_y);
							int numFeatures = 0;

							if (tp.size() <= start)
							{
								break;
							}
							const Template& templ = tp[start];
							numFeatures += static_cast<int>(templ.features.size());

							if (templ.features.size() < 64)
							{
								similarityLocal_64(lms[0], templ, similarities2, size, T, cv::Point(x, y));
								similarities2.convertTo(similarities2, CV_16U);
							}
							else if (templ.features.size() < 8192)
							{
								similarityLocal(lms[0], templ, similarities2, size, T, cv::Point(x, y));
							}
							else
							{
								qInfo() << ("feature size too large");
							}

							float best_score = 0;
							int best_r = -1, best_c = -1;
							for (int r = 0; r < similarities2.rows; ++r)
							{
								ushort* row = similarities2.ptr<ushort>(r);
								for (int c = 0; c < similarities2.cols; ++c)
								{
									int score_int = row[c];
									float score = (score_int * 100.f) / (4 * numFeatures);
									if (score > best_score)
									{
										best_score = score;
										best_r = r;
										best_c = c;
									}
								}
							}
							match2.similarity = best_score;
							match2.x = (x / T - 8 + best_c) * T + offset;
							match2.y = (y / T - 8 + best_r) * T + offset;
						}
						std::vector<Match>::iterator new_end = std::remove_if(candidates.begin(), candidates.end(), MatchPredicate(threshold));
						candidates.erase(new_end, candidates.end());
					}

					std::lock_guard<std::mutex> locker(_mutex);
					matches.insert(matches.end(), candidates.begin(), candidates.end());
				}
				});
		}
		catch (...)
		{
			qInfo() << ("  TM_CV matchClass  catch other error!!!");
		}
	}
#pragma endregion

	static std::mutex mtx; // 训练和匹配不同时
	bool Detector::trainTemplate(const cv::Mat& src, const std::string& id, const std::string& save_dirpath, TemplateParams params)
	{
		std::lock_guard<std::mutex> lck(mtx);

		if (src.empty() || id.empty())
		{
			return false;
		}

		int infos_idx = 0;
		res_map_mag_thresh = params.strong_thresh;
		this->modality = cv::makePtr<ColorGradient>(params.weak_thresh, params.num_features, params.strong_thresh, params.extract_kernel);
		this->setTatlevel(params.T);

		TemplateInfos info;
		info.template_id = id;
		info.params = params;
		if (this->template_infos.empty())
		{
			// id序列为空 直接存入
			this->template_infos.push_back(info);
			infos_idx = 0;
		}
		else
		{
			// id序列不为空，遍历id,存在则清空数据
			bool temp = false;
			for (auto i = 0; i < template_infos.size(); ++i)
			{
				auto& ti = template_infos[i];
				if (id == ti.template_id)
				{
					ti.shape_infos.clear();
					infos_idx = i;

					if (!class_templates.empty())
					{
						auto tm = class_templates.find(id);
						tm->second.clear();
					}
					temp = true;
					break;
				}
			}
			if (!temp)
			{
				this->template_infos.push_back(info);
				infos_idx = template_infos.size() - 1;
			}
		}

		if (id != this->template_infos[infos_idx].template_id)
		{
			return false;
		}

		auto& _Tinfo = this->template_infos[infos_idx];
		_Tinfo.image = src.clone();
		_Tinfo.mask = cv::Mat(src.size(), CV_8UC1, cv::Scalar::all(255));

		// 生成角度、尺度数据
		shape_info::ShapeInfo shapes(_Tinfo.image, _Tinfo.mask);
		shapes.angle_range = params.angle_range;
		shapes.angle_step = params.angle_step;
		shapes.scale_range = params.scale_range;
		shapes.scale_step = params.scale_step;
		shapes.produce_infos();

		// 生成模板数据
		_Tinfo.shape_infos.reserve(shapes.infos.size());
		this->addTemplate(shapes, id, _Tinfo.shape_infos);

		std::string filename_info = save_dirpath + "/infos_" + id + ".yaml";
		std::string filename_shape = save_dirpath + "/shape_" + id + ".yaml";
		cv::FileStorage fs(filename_info, cv::FileStorage::WRITE);
		writeClass(id, fs);
		shapes.save_infos(_Tinfo.shape_infos, filename_shape);

		return true;
	}

	//std::vector<MatchResults> Detector::matchTemplate(const cv::Mat& src, float match_threshold, std::vector<bool> status, cv::Mat& mat_out, std::vector<std::vector<cv::Point>>& vecpoints)
	//{
	//	std::lock_guard<std::mutex> lck2(mtx);

	//	if (src.empty() || 0 == template_infos.size() || 0 == pyramid_levels || 0 == class_templates.size() || template_infos.size() != status.size())
	//	{
	//		qWarning() << QString("训练失败,不进行检测!");
	//		return std::vector<MatchResults>();
	//	}

	//	// produce dxy, for icp purpose maybe
	//	set_produce_dxy = true;
	//	std::vector<MatchResults> resMatchs;

	//	try
	//	{
	//		std::vector<std::vector<Match>> mate = match(src, match_threshold, template_infos, status);

	//		src.copyTo(mat_out);
	//		if (3 != mat_out.channels())
	//		{
	//			mat_out = XZJToolBox::mat2Color(mat_out);
	//		}

	//		auto mSize = mate.size();
	//		vecpoints.reserve(mSize);
	//		for (auto t = 0; t < mSize; ++t)
	//		{
	//			if (template_infos.size() < mSize)
	//			{
	//				break;
	//			}
	//			auto info = template_infos[t];
	//			auto now_mate = mate[t];
	//			auto class_id = info.template_id;
	//			auto template_image = info.image;
	//			auto template_image_mask = info.mask;
	//			auto width = template_image.cols;
	//			auto height = template_image.rows;

	//			size_t draw_size = 1;
	//			draw_size = draw_size > now_mate.size() ? now_mate.size() : draw_size;

	//			std::vector<cv::Point> points;
	//			// 绘制
	//			for (size_t i = 0; i < draw_size; ++i)
	//			{
	//				auto match = now_mate[i];
	//				auto templ = getTemplates(class_id, match.template_id);
	//				auto info_good = info.shape_infos[match.template_id];

	//				for (int j = 0; j < templ[0].features.size(); ++j)
	//				{
	//					auto ft = templ[0].features[j];
	//					cv::Point pt_for_template(ft.x + match.x, ft.y + match.y);
	//					points.push_back(pt_for_template);
	//					cv::drawMarker(mat_out, pt_for_template, cv::Scalar(0, 255, 0), cv::MARKER_CROSS, 1);
	//				}

	//				std::vector<cv::Point> poly_pts;
	//				cv::Size rs_image_size;
	//				cv::Mat T_rs;
	//				XZJToolBox::getRotateTransform(template_image, info_good.angle, info_good.scale, rs_image_size, T_rs);
	//				int match_center_x = match.x + rs_image_size.width / 2 - templ[0].tl_x;
	//				int match_center_y = match.y + rs_image_size.height / 2 - templ[0].tl_y;

	//				// 记录匹配结果输出
	//				MatchResults matchRes;
	//				matchRes.angle = info_good.angle;
	//				matchRes.scale = info_good.scale;
	//				matchRes.center = cv::Point(match_center_x, match_center_y);
	//				matchRes.sTempID = class_id;
	//				matchRes.nMatchNum = now_mate.size();
	//				matchRes.dSimilarity = match.similarity;

	//				//cv::drawMarker(mat_out, matchRes.center, Scalar(0, 255, 255), MARKER_CROSS, 7);
	//				cv::Point region_center(width / 2, height / 2);
	//				cv::Mat T_0 = XZJToolBox::getTransform(info_good.angle, info_good.scale, match_center_x, match_center_y);

	//				cv::Mat T_pixel2center = cv::Mat::zeros(3, 3, CV_64FC1);
	//				T_pixel2center.at<double>(0, 0) = 1;
	//				T_pixel2center.at<double>(0, 2) = -region_center.x;
	//				T_pixel2center.at<double>(1, 2) = -region_center.y;
	//				T_pixel2center.at<double>(1, 1) = 1;
	//				T_pixel2center.at<double>(2, 2) = 1;
	//				cv::Mat T_center2base;
	//				cv::invert(T_pixel2center, T_center2base);

	//				cv::Mat T_base2image, T_0_inv, T_temp;
	//				cv::invert(T_0, T_0_inv);
	//				cv::invert(T_center2base * T_0_inv, T_base2image);
	//				cv::Mat T_left_corner2center = cv::Mat::zeros(3, 3, CV_64FC1);
	//				T_left_corner2center.at<double>(0, 0) = 1;
	//				T_left_corner2center.at<double>(0, 2) = -width / 2;
	//				T_left_corner2center.at<double>(1, 2) = -height / 2;
	//				T_left_corner2center.at<double>(1, 1) = 1;
	//				T_left_corner2center.at<double>(2, 2) = 1;

	//				cv::Mat T_left_corner2image = T_base2image * (T_center2base * T_left_corner2center);
	//				cv::Mat train_area_wh_temp0 = cv::Mat::zeros(3, 1, CV_64FC1);
	//				train_area_wh_temp0.at<double>(0, 0) = width;
	//				cv::Mat train_area_wh_temp1 = cv::Mat::zeros(3, 1, CV_64FC1);
	//				train_area_wh_temp1.at<double>(1, 0) = height;
	//				cv::Mat mul_m1 = T_left_corner2image * train_area_wh_temp0;
	//				double train_area_w = norm(T_left_corner2image * train_area_wh_temp0, cv::NORM_L2) / info_good.scale;
	//				double train_area_h = norm(T_left_corner2image * train_area_wh_temp1, cv::NORM_L2) / info_good.scale;

	//				std::vector<cv::Point> polySrc = { cv::Point(0,0) + matchRes.center - region_center,		cv::Point(train_area_w,0) + matchRes.center - region_center,
	//							cv::Point(train_area_w,train_area_h) + matchRes.center - region_center, cv::Point(0,train_area_h) + matchRes.center - region_center };
	//				//cv::polylines(mat_out, polySrc, true, cv::Scalar(125, 125, 0), 1);

	//				cv::Mat M_Temp = getRotationMatrix2D(matchRes.center, -matchRes.angle, 1);
	//				auto ptssss = XZJToolBox::transfromPts(polySrc, M_Temp);
	//				//cv::polylines(mat_out, ptssss, true, cv::Scalar(255, 255, 0), 2);
	//				matchRes.ptsBox = ptssss;
	//				resMatchs.push_back(matchRes);
	//			}

	//			vecpoints.emplace_back(points);
	//		}
	//	}
	//	catch (...)
	//	{
	//		qInfo() << (" matchTemplate catch other error!!!");
	//		return resMatchs;
	//	}

	//	return resMatchs;
	//}

	bool Detector::setTatlevel(int level)
	{
		level = level < 1 ? 1 : level;
		T_at_level.clear();
		T_at_level.reserve(level);
		pyramid_levels = level;

		for (auto i = 0; i < level; ++i)
		{
			auto _i = i < 2 ? i : 1;
			auto _t = 2 << (_i + 1);
			T_at_level.push_back(_t);
		}
		return true;
	}

	Detector::TemplateParams Detector::getParam(std::string id) const
	{
		for (auto tpinfo : template_infos)
		{
			if (id == tpinfo.template_id)
			{
				return tpinfo.params;
			}
		}
		return TemplateParams();
	}

	bool Detector::deleteModelData(const std::string& id)
	{
		if (id.empty())
		{
			return false;
		}

		for (auto it = template_infos.begin(); it != template_infos.end();)
		{
			if (id == (*it).template_id)
				it = template_infos.erase(it);
			else
				++it;
		}
		class_templates.erase(id);
		return true;
	}

	bool Detector::deleteAllModelData()
	{
		if (!template_infos.empty())
		{
			template_infos.clear();
		}
		if (!class_templates.empty())
		{
			class_templates.clear();
		}

		return true;
	}

} // namespace line2Dup
