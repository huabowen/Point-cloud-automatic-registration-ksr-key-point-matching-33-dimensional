#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <vector>
#include <string>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/features/fpfh.h>
#include <pcl/features/pfh.h>
#include <pcl/visualization/pcl_plotter.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/approximate_voxel_grid.h>
#include <pcl/recognition/cg/geometric_consistency.h>
#include <pcl/common/transforms.h>
#include <pcl/recognition/cg/hough_3d.h>
#include <pcl/features/board.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/sample_consensus_prerejective.h>
#include <pcl/registration/correspondence_rejection_one_to_one.h>
#include <math.h>
#include <pcl/keypoints/uniform_sampling.h>
#include <pcl/features/pfh.h>
#include <thread>
#include<mutex>

using namespace std;  // 可以加入 std 的命名空间
mutex m;
struct i_p_t {
	int i = 0;
	float percent = 0.0f;
	Eigen::Matrix4f trans;
};
struct c_k_f {
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud;
	pcl::PointCloud<pcl::PointXYZ>::Ptr key;
	pcl::PointCloud<pcl::FPFHSignature33>::Ptr features;
};
vector<c_k_f> clouds_keys_features;
vector<i_p_t> results;
///////////////////显示//////////////////////////////////////////////////////////////////////////////
void show_point_cloud(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud) {
	pcl::visualization::PCLVisualizer viewer_final("3D Viewer");
	viewer_final.setBackgroundColor(255, 255, 255); //白色背景
	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_cloud(cloud, 0, 255, 0);//蓝色点云
	viewer_final.addPointCloud<pcl::PointXYZ>(cloud, color_cloud, "1");
	while (!viewer_final.wasStopped()) {
		viewer_final.spinOnce(100);
		boost::this_thread::sleep(boost::posix_time::microseconds(100000));
	}
}

void show_point_clouds(vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> clouds) {
	pcl::visualization::PCLVisualizer viewer_final("3D Viewer");
	viewer_final.setBackgroundColor(255, 255, 255);   //白色背景
	for (int i = 0; i < clouds.size(); i++) {
		pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color(clouds[i], rand() % 255, rand() % 255, rand() % 255);
		viewer_final.addPointCloud<pcl::PointXYZ>(clouds[i], color, to_string(i));
	}
	while (!viewer_final.wasStopped()) {
		viewer_final.spinOnce(100);
		boost::this_thread::sleep(boost::posix_time::microseconds(100000));
	}
}

void show_key_point(pcl::PointCloud<pcl::PointXYZ>::Ptr Acloud, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_keypoint) {
	pcl::visualization::PCLVisualizer viewer_final("3D Viewer");
	viewer_final.setBackgroundColor(255, 255, 255);  //白色背景
	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_cloud(Acloud, 0, 255, 0);//蓝色点云
	viewer_final.addPointCloud<pcl::PointXYZ>(Acloud, color_cloud, "1");
	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_key(cloud_keypoint, 255, 0, 0);//关键点
	viewer_final.addPointCloud<pcl::PointXYZ>(cloud_keypoint, color_key, "2");
	viewer_final.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "2");
	while (!viewer_final.wasStopped()) {
		viewer_final.spinOnce(100);
		boost::this_thread::sleep(boost::posix_time::microseconds(100000));
	}
}

void show_key_corr(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_source, pcl::PointCloud<pcl::PointXYZ>::Ptr key_source,
	pcl::PointXYZ ps0, pcl::PointXYZ ps1, pcl::PointXYZ ps2,
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_target, pcl::PointCloud<pcl::PointXYZ>::Ptr key_target,
	pcl::PointXYZ pt0, pcl::PointXYZ pt1, pcl::PointXYZ pt2) {

	pcl::PointCloud<pcl::PointXYZ>::Ptr key_ps0(new pcl::PointCloud<pcl::PointXYZ>);
	key_ps0->push_back(ps0);
	pcl::PointCloud<pcl::PointXYZ>::Ptr key_ps(new pcl::PointCloud<pcl::PointXYZ>);
	key_ps->push_back(ps1);
	key_ps->push_back(ps2);
	pcl::PointCloud<pcl::PointXYZ>::Ptr key_pt0(new pcl::PointCloud<pcl::PointXYZ>);
	key_pt0->push_back(pt0);
	pcl::PointCloud<pcl::PointXYZ>::Ptr key_pt(new pcl::PointCloud<pcl::PointXYZ>);
	key_pt->push_back(pt1);
	key_pt->push_back(pt2);
	pcl::visualization::PCLVisualizer view("3D Viewer");
	int v1(0);
	int v2(1);
	view.createViewPort(0.0, 0.0, 0.5, 1.0, v1);
	view.createViewPort(0.5, 0.0, 1.0, 1.0, v2);
	view.setBackgroundColor(255, 255, 255, v1);
	view.setBackgroundColor(255, 255, 255, v2);
	view.addPointCloud<pcl::PointXYZ>(cloud_source, pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>(cloud_source, 0, 0, 0), "cloud_source", v1);
	view.addPointCloud<pcl::PointXYZ>(key_ps0, pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>(key_ps0, 255, 0, 0), "key_ps0", v1);
	view.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "key_ps0", v1);
	view.addPointCloud<pcl::PointXYZ>(key_ps, pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>(key_ps, 0, 255, 0), "key_ps", v1);
	view.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "key_ps", v1);
	view.addPointCloud<pcl::PointXYZ>(cloud_target, pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>(cloud_target, 0, 0, 0), "cloud_target", v2);
	view.addPointCloud<pcl::PointXYZ>(key_pt0, pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>(key_pt0, 255, 0, 0), "key_pt0", v2);
	view.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "key_pt0", v2);
	view.addPointCloud<pcl::PointXYZ>(key_pt, pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>(key_pt, 0, 255, 0), "key_pt", v2);
	view.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "key_pt", v2);
	while (!view.wasStopped()) {
		view.spinOnce(100);
		boost::this_thread::sleep(boost::posix_time::microseconds(100000));
	}
}

void show_coor(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_model, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_scenes, pcl::PointCloud<pcl::PointXYZ> keypoints_model, pcl::PointCloud<pcl::PointXYZ> keypoints_scenes,
	pcl::PointCloud<pcl::FPFHSignature33>::Ptr features_model, pcl::PointCloud<pcl::FPFHSignature33>::Ptr features_scenes, pcl::CorrespondencesPtr corr) {
	for (int i = 0; i < corr->size(); i++) {
		cout << corr->at(i).index_query << "---" << corr->at(i).index_match << "---" << corr->at(i).distance << endl;
		pcl::visualization::PCLPlotter plotter;
		plotter.addFeatureHistogram<pcl::FPFHSignature33>(*features_model, "fpfh", corr->at(i).index_query);
		plotter.addFeatureHistogram<pcl::FPFHSignature33>(*features_scenes, "fpfh", corr->at(i).index_match);
		//cout << features_model->points[corr->at(i).index_query] << endl;
		//cout << features_scenes->points[corr->at(i).index_match] << endl;
		pcl::PointCloud<pcl::PointXYZ>::Ptr keypoints_ptr_model(new pcl::PointCloud<pcl::PointXYZ>());
		pcl::PointCloud<pcl::PointXYZ>::Ptr keypoints_ptr_scenes(new pcl::PointCloud<pcl::PointXYZ>());
		keypoints_ptr_model->push_back(keypoints_model.points[corr->at(i).index_query]);
		keypoints_ptr_scenes->push_back(keypoints_scenes.points[corr->at(i).index_match]);
		boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
		int v1(0);
		viewer->createViewPort(0.0, 0.0, 0.5, 1.0, v1);  //4个参数分别是X轴的最小值，最大值，Y轴的最小值，最大值，取值0-1，v1是标识
		viewer->setBackgroundColor(255, 255, 255, v1);    //设置视口的背景颜色
		pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_key_model(keypoints_ptr_model, 255, 0, 0);
		viewer->addPointCloud<pcl::PointXYZ>(keypoints_ptr_model, color_key_model, "color_key_model", v1);
		viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 10, "color_key_model");
		pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_cloud_model(cloud_model, 0, 0, 255);
		viewer->addPointCloud<pcl::PointXYZ>(cloud_model, color_cloud_model, "cloud_model", v1);
		int v2(0);
		viewer->createViewPort(0.5, 0.0, 1.0, 1.0, v2);  //4个参数分别是X轴的最小值，最大值，Y轴的最小值，最大值，取值0-1，v1是标识
		viewer->setBackgroundColor(255, 255, 255, v2);    //设置视口的背景颜色
		pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_key_scenes(keypoints_ptr_scenes, 255, 0, 0);
		viewer->addPointCloud<pcl::PointXYZ>(keypoints_ptr_scenes, color_key_scenes, "color_key_scenes", v2);
		viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 10, "color_key_scenes");
		pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_cloud_scenes(cloud_scenes, 0, 0, 255);
		viewer->addPointCloud<pcl::PointXYZ>(cloud_scenes, color_cloud_scenes, "cloud_scenes", v2);
		plotter.plot();
		// 等待直到可视化窗口关闭
		while (!viewer->wasStopped()) {
			viewer->spinOnce(100);
			//boost::this_thread::sleep(boost::posix_time::microseconds(100000));
		}
	}
}

void show_line(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_source, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_target, pcl::PointCloud<pcl::PointXYZ>::Ptr key_source,
	pcl::PointCloud<pcl::PointXYZ>::Ptr key_target, pcl::CorrespondencesPtr corr, float leaf_size) {

	pcl::PointCloud<pcl::PointXYZ>::Ptr new_key_source(new pcl::PointCloud<pcl::PointXYZ>());
	pcl::PointCloud<pcl::PointXYZ>::Ptr new_key_target(new pcl::PointCloud<pcl::PointXYZ>());
	for (int i = 0; i < corr->size(); i++) {
		new_key_source->push_back(key_source->points[corr->at(i).index_query]);
		new_key_target->push_back(key_target->points[corr->at(i).index_match]);
	}

	pcl::PointCloud<pcl::PointXYZ>::Ptr new_cloud_source(new pcl::PointCloud<pcl::PointXYZ>);
	*new_cloud_source = *cloud_source;
	for (int i = 0; i < cloud_source->size(); i++) {
		new_cloud_source->points[i].y += 300.0f* leaf_size;
	}
	for (int i = 0; i < new_key_source->size(); i++) {
		new_key_source->points[i].y += 300.0f* leaf_size;
	}
	pcl::visualization::PCLVisualizer line("line");
	line.setBackgroundColor(255, 255, 255);
	line.addPointCloud(cloud_target, pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>(cloud_target, 0.0, 0, 255.0), "cloud_target");
	line.addPointCloud(new_cloud_source, pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>(new_cloud_source, 0.0, 255, 0), "cloud_source");
	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>color_new_key_target(new_key_target, 255, 0, 0);//红色关键点
	line.addPointCloud<pcl::PointXYZ>(new_key_target, color_new_key_target, "new_key_target");
	line.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 4, "new_key_target");

	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_new_key_source(new_key_source, 255, 0, 0);//红色关键点
	line.addPointCloud<pcl::PointXYZ>(new_key_source, color_new_key_source, "new_key_source");
	line.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 4, "new_key_source");

	for (int i = 0; i < new_key_source->size(); i++) {
		pcl::PointXYZ source_point = new_key_source->points[i];
		pcl::PointXYZ target_point = new_key_target->points[i];
		line.addLine(source_point, target_point, 255, 0, 255, to_string(i));
		line.setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 2, to_string(i));
	}
	line.spin();
}

void show_point_clouds_and_trans_models(vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> scenes, vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> models, vector<i_p_t> result_final) {

	vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> models_results;
	pcl::visualization::PCLVisualizer viewer_final("3D Viewer");
	viewer_final.setBackgroundColor(255, 255, 255);   //白色背景
	for (int i = 0; i < scenes.size(); i++) {
		pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color(scenes[i], rand() % 255, rand() % 255, rand() % 255);
		viewer_final.addPointCloud<pcl::PointXYZ>(scenes[i], color, "scenes" + to_string(i));
		pcl::PointCloud<pcl::PointXYZ>::Ptr models_result(new pcl::PointCloud<pcl::PointXYZ>);
		*models_result = *models[result_final[i].i];
		pcl::transformPointCloud(*models_result, *models_result, result_final[i].trans);
		models_results.push_back(models_result);
		if (result_final[i].percent < 0.8f)
			continue;
		pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_model(models_results[i], rand() % 255, rand() % 255, rand() % 255);
		viewer_final.addPointCloud<pcl::PointXYZ>(models_results[i], color_model, "models_results" + to_string(i));
	}
	while (!viewer_final.wasStopped()) {
		viewer_final.spinOnce(100);
		boost::this_thread::sleep(boost::posix_time::microseconds(100000));
	}
}


/////////////////循环滤波///////////////////////////////////////////////////////////////////////////////////////////////////////
pcl::PointCloud<pcl::Normal>::Ptr normal_estimation_OMP(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, float leaf_size) {
	pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>());
	pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::Normal> ne;
	ne.setNumberOfThreads(10);
	ne.setInputCloud(cloud);
	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
	ne.setSearchMethod(tree);
	ne.setRadiusSearch(leaf_size);
	ne.compute(*normals);
	return normals;
}

float com_leaf(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud) {
	pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
	kdtree.setInputCloud(cloud);//Acloud在Bcloud中进行搜索
	int K = 2;
	std::vector<int> pointIdxNKNSearch(K);//最近点索引
	std::vector<float> pointNKNSquaredDistance(K);//最近点距离
	float leaf_size = 0;
	for (int i = 0; i < cloud->size(); i++) {
		kdtree.nearestKSearch(cloud->points[i], K, pointIdxNKNSearch, pointNKNSquaredDistance);
		leaf_size = leaf_size + sqrt(pointNKNSquaredDistance[1]);
		pointIdxNKNSearch.clear();
		pointNKNSquaredDistance.clear();
	}
	leaf_size = (float)leaf_size / (float)(cloud->size());
	//cout << "平均距离：" << leaf_size << endl;
	return leaf_size;
}

float get_leaf_size_by_nums(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, int nums, float leaf_size) {
	return leaf_size * pow((float)cloud->size() / (float)nums, 1.0 / 3.0);
}

float get_leaf_size_by_leaf_size(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, float leaf_size, float source_leaf_size) {
	return source_leaf_size + 0.2f*(source_leaf_size - leaf_size);
}

pcl::PointCloud<pcl::PointXYZ>::Ptr voxel_grid(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, float leaf_size) {
	//体素滤波
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::VoxelGrid<pcl::PointXYZ> sor;  //创建滤波对象
	sor.setInputCloud(cloud);            //设置需要过滤的点云给滤波对象
	sor.setLeafSize(leaf_size, leaf_size, leaf_size);  //设置滤波时创建的体素体积
	sor.filter(*cloud_filtered);           //执行滤波处理，存储输出	
	return cloud_filtered;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud_to_num(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, int nums = 20000) {
	float leaf_size = com_leaf(cloud);
	while (cloud->points.size() > nums + nums * 0.2f) {
		leaf_size = get_leaf_size_by_nums(cloud, nums, leaf_size);
		*cloud = *voxel_grid(cloud, leaf_size);
		leaf_size = com_leaf(cloud);
		cout << "点云点数：" << cloud->size() << endl;
	}
	return cloud;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud_to_leaf_size(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, float source_leaf_size) {
	float leaf_size = com_leaf(cloud);
	while (source_leaf_size > leaf_size + leaf_size * 0.02f) {
		leaf_size = get_leaf_size_by_leaf_size(cloud, leaf_size, source_leaf_size);
		*cloud = *voxel_grid(cloud, leaf_size);
		leaf_size = com_leaf(cloud);
		cout << "点云点数：" << cloud->size() << endl;
	}
	return cloud;
}


///////////////////////关键点查找//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
float com_avg_curvature(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, pcl::PointCloud<pcl::Normal>::Ptr normals, int i, float radius, pcl::search::KdTree<pcl::PointXYZ>::Ptr tree) {
	float avg_curvature = 0;
	vector<int> point_ind;
	vector<float> point_dist;
	tree->radiusSearch(cloud->points[i], radius, point_ind, point_dist);
	for (int i = 0; i < point_ind.size(); i++) {
		avg_curvature += normals->points[point_ind[i]].curvature;
	}
	avg_curvature = avg_curvature / float(point_ind.size());
	return avg_curvature;
}

bool is_max_avg_curvature(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, vector<float> avg_curvatures, int point, vector<bool>& possible_key, vector<bool> possible_key_possible,
	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree, float radius = 4) {

	vector<int> point_ind;
	vector<float> point_dis;
	tree->radiusSearch(cloud->points[point], radius, point_ind, point_dis);//此处半径为计算曲率和的半径
	if (point_ind.size() < 5)
		return false;
	for (int i = 1; i < point_ind.size(); i++) {
		if (possible_key_possible[point_ind[i]]) {
			if (avg_curvatures[point_ind[0]] > avg_curvatures[point_ind[i]])
				possible_key[point_ind[i]] = false;
			else if (avg_curvatures[point_ind[0]] < avg_curvatures[point_ind[i]])
				possible_key[point_ind[0]] = false;
		}

	}
	return possible_key[point_ind[0]];
}

pcl::PointCloud<pcl::PointXYZ>::Ptr key_detect(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, pcl::PointCloud<pcl::Normal>::Ptr normals, pcl::search::KdTree<pcl::PointXYZ>::Ptr tree,
	float radius_curvature = 5) {

	vector<bool> possible_key(cloud->size(), false);
	//int nums = 0;
	for (int i = 0; i < cloud->size(); i++) {
		if (normals->points[i].curvature > 0.01) {
			possible_key[i] = true;
			//nums += 1;
		}
	}
	vector<bool> possible_key_possible(possible_key);
	vector<float> avg_curvatures;
	for (int i = 0; i < cloud->size(); i++) {
		if (possible_key[i])
			avg_curvatures.push_back(com_avg_curvature(cloud, normals, i, radius_curvature, tree));
		else
			avg_curvatures.push_back(0);
	}
	pcl::PointCloud<pcl::PointXYZ>::Ptr key(new pcl::PointCloud<pcl::PointXYZ>);
	for (int i = 0; i < cloud->size(); i++) {
		if (possible_key[i]) {
			if (is_max_avg_curvature(cloud, avg_curvatures, i, possible_key, possible_key_possible, tree, radius_curvature)) {//此处半径为计算曲率和的半径
				key->push_back(cloud->points[i]);
			}
		}
	}
	return key;
}

///////////////////////对比实验关键点查找//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
pcl::PointCloud<pcl::PointXYZ>::Ptr key_detect_u(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, float leaf_size) {
	pcl::PointCloud<pcl::PointXYZ>::Ptr key(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::UniformSampling<pcl::PointXYZ> filter;
	filter.setInputCloud(cloud);
	filter.setRadiusSearch(leaf_size);
	filter.filter(*key);
	return key;
}


/////////////////////特征描述/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
float com_angle(float nx, float ny, float nz, float cx, float cy, float cz) {
	if ((cx == 0 && cy == 0 && cz == 0) || isnan(nx) || isnan(ny) || isnan(nz))
		return 0;
	float angle = acos((nx*cx + ny * cy + nz * cz) / (sqrt(pow(nx, 2) + pow(ny, 2) + pow(nz, 2))*sqrt(pow(cx, 2) + pow(cy, 2) + pow(cz, 2))))*(180.0f / 3.1415f);
	return angle;
}

float com_dist(float nx, float ny, float nz, float cx, float cy, float cz) {
	float distance = 0;
	distance = sqrt(pow(nx - cx, 2) + pow(ny - cy, 2) + pow(nz - cz, 2));
	return distance;
}

pcl::PointCloud<pcl::FPFHSignature33>::Ptr com_features(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, pcl::PointCloud<pcl::Normal>::Ptr normals,
	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree, pcl::PointCloud<pcl::PointXYZ>::Ptr key, float radius, int num_distance = 15, int num_angle = 18) {

	pcl::PointCloud<pcl::FPFHSignature33>::Ptr features(new pcl::PointCloud<pcl::FPFHSignature33>());
	for (int k = 0; k < key->size(); k++) {
		vector<int> indices;
		vector<float> dists;
		tree->radiusSearch(key->points[k], radius, indices, dists);
		pcl::FPFHSignature33 feature;
		for (int i = 0; i < feature.descriptorSize(); i++) {
			feature.histogram[i] = 0;
		}
		pcl::PointCloud<pcl::PointXYZ>::Ptr search_points(new pcl::PointCloud<pcl::PointXYZ>());
		pcl::PointCloud<pcl::Normal>::Ptr search_normals(new pcl::PointCloud<pcl::Normal>());
		pcl::copyPointCloud(*cloud, indices, *search_points);
		pcl::copyPointCloud(*normals, indices, *search_normals);
		Eigen::Vector4f centroid;  //质心 
		pcl::compute3DCentroid(*search_points, centroid); //估计质心的坐标
		//cout << centroid.x() << " " << centroid.y() << " " << centroid.z() << endl;
		vector<float> vec_distance;
		for (int i = 0; i < search_points->size(); i++) {
			float distance = 0;
			distance = com_dist(search_points->points[i].x, search_points->points[i].y, search_points->points[i].z, centroid.x(), centroid.y(), centroid.z());
			vec_distance.push_back(distance);
		}
		float max_diatance = *std::max_element(std::begin(vec_distance), std::end(vec_distance));
		float min_diatance = *std::min_element(std::begin(vec_distance), std::end(vec_distance));
		float res_distance = (max_diatance - min_diatance) / num_distance;
		for (int i = 0; i < search_points->size(); i++) {
			float angle = com_angle(search_normals->points[i].normal_x, search_normals->points[i].normal_y, search_normals->points[i].normal_z,
				search_points->points[i].x - centroid.x(), search_points->points[i].y - centroid.y(), search_points->points[i].z - centroid.z());
			int bin_angle = int(angle / 10.0f);
			int bin_distance = 0;
			if (res_distance != 0) {
				bin_distance = int((vec_distance[i] - min_diatance) / res_distance);
			}
			if (bin_distance > num_distance - 1) bin_distance = num_distance - 1;
			if (bin_angle > num_angle - 1) bin_angle = num_angle - 1;
			//feature.histogram[bin_distance] += 1;
			feature.histogram[bin_distance] += 1;
			feature.histogram[num_distance + bin_angle] += 1;
		}
		for (int i = 0; i < feature.descriptorSize(); i++) {
			feature.histogram[i] = feature.histogram[i] / ((float)search_points->size());
		}
		features->push_back(feature);
		//pcl::visualization::PCLPlotter plotter;
		//plotter.addFeatureHistogram<pcl::FPFHSignature33>(*features, "fpfh", k);
		//plotter.plot();
	}
	return features;
}


/////////////////////对比实验特征描述/////////////////////////////////////////////////////////////////////////////////////////////////////
pcl::PointCloud<pcl::PFHSignature125>::Ptr com_pfh_feature(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, pcl::PointCloud<pcl::Normal>::Ptr normal, pcl::PointCloud<pcl::PointXYZ>::Ptr key, float leaf_size) {
	pcl::PointCloud<pcl::PFHSignature125>::Ptr features(new pcl::PointCloud<pcl::PFHSignature125>());
	pcl::PFHEstimation<pcl::PointXYZ, pcl::Normal, pcl::PFHSignature125> pfh;
	pfh.setInputCloud(key);
	pfh.setInputNormals(normal);
	pfh.setSearchSurface(cloud);
	pfh.setRadiusSearch(leaf_size);
	pfh.compute(*features);
	return features;
}


/////////////////////特征匹配/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
float com_angle(pcl::PointXYZ p, pcl::PointXYZ p1, pcl::PointXYZ p2) {
	return acos(((p1.x - p.x)*(p2.x - p.x) + (p1.y - p.y)*(p2.y - p.y) + (p1.z - p.z)*(p2.z - p.z)) / (sqrt(pow(p1.x - p.x, 2) + pow(p1.y - p.y, 2) + pow(p1.z - p.z, 2))*sqrt(pow(p2.x - p.x, 2) + pow(p2.y - p.y, 2) + pow(p2.z - p.z, 2))))*(180.0f / 3.1415f);
}

float com_feature_dist(pcl::FPFHSignature33 f1, pcl::FPFHSignature33 f2) {
	float dist = 0.0f;
	for (int i = 0; i < f1.descriptorSize(); i++) {
		dist += pow(f1.histogram[i] - f2.histogram[i], 2);
	}
	return sqrt(dist);
}

pcl::CorrespondencesPtr com_correspondence(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_source, pcl::PointCloud<pcl::PointXYZ>::Ptr key_source, pcl::PointCloud<pcl::FPFHSignature33>::Ptr feature_source,
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_target, pcl::PointCloud<pcl::PointXYZ>::Ptr key_target, pcl::PointCloud<pcl::FPFHSignature33>::Ptr feature_target, float dis, float leaf_size) {
	pcl::CorrespondencesPtr corrs(new pcl::Correspondences());
	pcl::KdTreeFLANN<pcl::FPFHSignature33> feature_target_kdtree;
	feature_target_kdtree.setInputCloud(feature_target);
	pcl::KdTreeFLANN<pcl::PointXYZ> key_source_kdtree;
	key_source_kdtree.setInputCloud(key_source);
	pcl::KdTreeFLANN<pcl::PointXYZ> key_target_kdtree;
	key_target_kdtree.setInputCloud(key_target);
	for (size_t i = 0; i < key_source->size(); i++) {
		std::vector<int> key_source_indices;
		std::vector<float> key_source_dists;
		key_source_kdtree.nearestKSearch(key_source->points[i], 3, key_source_indices, key_source_dists);
		float angle1 = com_angle(key_source->points[i], key_source->points[key_source_indices[1]], key_source->points[key_source_indices[2]]);
		std::vector<int> corr_indices;
		std::vector<float> corr_dists;
		feature_target_kdtree.nearestKSearch(feature_source->points[i], 20, corr_indices, corr_dists);
		if (sqrt(corr_dists[0]) < 0.5*sqrt(corr_dists[1])) {
			pcl::Correspondence corr0(i, corr_indices[0], sqrt(corr_dists[0]));
			corrs->push_back(corr0);
			continue;
		}
		for (int j = 0; j < corr_indices.size(); j++) {
			if (sqrt(corr_dists[j]) < dis) {
				std::vector<int> key_target_indices;
				std::vector<float> key_target_dists;
				key_target_kdtree.nearestKSearch(key_target->points[corr_indices[j]], 3, key_target_indices, key_target_dists);
				float angle2 = com_angle(key_target->points[corr_indices[j]], key_target->points[key_target_indices[1]], key_target->points[key_target_indices[2]]);
				float feature_dist1 = com_feature_dist(feature_source->points[key_source_indices[1]], feature_target->points[key_target_indices[1]]);
				float feature_dist2 = com_feature_dist(feature_source->points[key_source_indices[2]], feature_target->points[key_target_indices[2]]);
				float dist1 = abs(sqrt(key_source_dists[1]) - sqrt(key_target_dists[1]));
				float dist2 = abs(sqrt(key_source_dists[2]) - sqrt(key_target_dists[2]));
				float dist_angle = abs(angle1 - angle2);
				//cout << "dist1:" << dist1 << endl;
				//cout << "dist2:" << dist2 << endl;
				//cout << "feature_dist0:" << sqrt(corr_dists[j]) << endl;
				//cout << "feature_dist1:" << feature_dist1 << endl;
				//cout << "feature_dist2:" << feature_dist2 << endl;
				//cout << "dist_angle:" << dist_angle << endl;
				/*show_key_corr(cloud_source, key_source,
					key_source->points[key_source_indices[0]], key_source->points[key_source_indices[1]], key_source->points[key_source_indices[2]],
					cloud_target, key_target,
					key_target->points[key_target_indices[0]], key_target->points[key_target_indices[1]], key_target->points[key_target_indices[2]]);*/
				if (dist1 < 3.0f* leaf_size&&dist2 < 3.0f* leaf_size&&feature_dist1 < dis&&feature_dist2 < dis&&dist_angle < 10.0f) {
					//show_key_corr(cloud_source, key_source,
					//	key_source->points[key_source_indices[0]], key_source->points[key_source_indices[1]], key_source->points[key_source_indices[2]],
					//	cloud_target, key_target,
					//	key_target->points[key_target_indices[0]], key_target->points[key_target_indices[1]], key_target->points[key_target_indices[2]]);
					pcl::Correspondence corr(i, corr_indices[j], sqrt(corr_dists[j]));
					corrs->push_back(corr);
					pcl::Correspondence corr1(key_source_indices[1], key_target_indices[1], feature_dist1);
					corrs->push_back(corr1);
					pcl::Correspondence corr2(key_source_indices[2], key_target_indices[2], feature_dist2);
					corrs->push_back(corr2);
					//break;
				}
			}
			else break;
		}
	}
	return corrs;
}

////////////////////对比实验特征匹配/////////////////////////////////////////////////////////////////////////////////////
pcl::CorrespondencesPtr com_correspondence(pcl::PointCloud<pcl::PFHSignature125>::Ptr feature_source, pcl::PointCloud<pcl::PFHSignature125>::Ptr feature_target, float dis) {
	pcl::CorrespondencesPtr model_scene_corrs(new pcl::Correspondences());
	pcl::KdTreeFLANN<pcl::PFHSignature125> match_search;   //设置配准的方法
	match_search.setInputCloud(feature_target);  //输入模板点云的描述子
	for (size_t i = 0; i < feature_source->size(); ++i) {
		std::vector<int> neigh_indices(1);   //设置最近邻点的索引
		std::vector<float> neigh_sqr_dists(1); //申明最近邻平方距离值
		int found_neighs = match_search.nearestKSearch(feature_source->at(i), 1, neigh_indices, neigh_sqr_dists);
		if (found_neighs == 1 && sqrt(neigh_sqr_dists[0]) < dis) {
			pcl::Correspondence corr(static_cast<int> (i), neigh_indices[0], sqrt(neigh_sqr_dists[0]));
			model_scene_corrs->push_back(corr);   //把配准的点存储在容器中
		}
	}
	return model_scene_corrs;
}

///////////////////点云加噪声////////////////////////////////////
pcl::PointCloud<pcl::PointXYZ>::Ptr add_gaussian_noise(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, float m) {
	float leaf_size = 0;
	leaf_size = com_leaf(cloud);
	//添加高斯噪声
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloudfiltered(new pcl::PointCloud<pcl::PointXYZ>());
	cloudfiltered->points.resize(cloud->points.size());//将点云的cloud的size赋值给噪声
	cloudfiltered->header = cloud->header;
	cloudfiltered->width = cloud->width;
	cloudfiltered->height = cloud->height;
	boost::mt19937 rng;
	rng.seed(static_cast<unsigned int>(time(0)));
	boost::normal_distribution<> nd(0, m*leaf_size);
	boost::variate_generator<boost::mt19937&, boost::normal_distribution<>> var_nor(rng, nd);
	//添加噪声
	for (size_t point_i = 0; point_i < cloud->points.size(); ++point_i)
	{
		//cloudfiltered->points[point_i].x = cloud->points[point_i].x + static_cast<float> (var_nor());
		//cloudfiltered->points[point_i].y = cloud->points[point_i].y + static_cast<float> (var_nor());
		//cloudfiltered->points[point_i].z = cloud->points[point_i].z + static_cast<float> (var_nor());
		cloudfiltered->points[point_i].x = cloud->points[point_i].x + static_cast<float> (var_nor());
		cloudfiltered->points[point_i].y = cloud->points[point_i].y;
		cloudfiltered->points[point_i].z = cloud->points[point_i].z;
	}
	return cloudfiltered;
}

//////////////////欧式分割//////////////////////////
vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> euclidean_cluster(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, int tolerance = 4, int min = 1000, int max = 50000) {
	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
	tree->setInputCloud(cloud);
	std::vector<pcl::PointIndices> cluster_indices;
	pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;   //欧式聚类对象
	ec.setClusterTolerance(tolerance);                     // 设置近邻搜索的搜索半径为2cm
	ec.setMinClusterSize(min);                 //设置一个聚类需要的最少的点数目为100
	ec.setMaxClusterSize(max);               //设置一个聚类需要的最大点数目为25000
	ec.setSearchMethod(tree);                    //设置点云的搜索机制
	ec.setInputCloud(cloud);
	ec.extract(cluster_indices);           //从点云中提取聚类，并将点云索引保存在cluster_indices中
	vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> clouds;
	for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin(); it != cluster_indices.end(); ++it) {
		pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cluster(new pcl::PointCloud<pcl::PointXYZ>);
		for (std::vector<int>::const_iterator pit = it->indices.begin(); pit != it->indices.end(); ++pit)
			cloud_cluster->points.push_back(cloud->points[*pit]);
		cloud_cluster->width = cloud_cluster->points.size();
		cloud_cluster->height = 1;
		cloud_cluster->is_dense = true;
		clouds.push_back(cloud_cluster);
	}
	return clouds;
}

//////////////////计算重叠率////////////////////////
float com_overlap_rate(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_source, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_target, float leaf_size) {
	pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
	kdtree.setInputCloud(cloud_target);
	int count = 0;//重合点个数
	for (int i = 0; i < cloud_source->size(); i++) {
		std::vector<int> indices;
		std::vector<float> dists;
		if (kdtree.nearestKSearch(cloud_source->points[i], 1, indices, dists) > 0) {
			if (sqrt(dists[0]) < 5.0f*leaf_size)
				count++;
		}
	}
	float overlap_rate = (float)count / (float)cloud_source->size();
	//cout << "重叠率：" << overlap_rate * 100.0f << "%" << endl;
	return overlap_rate;
}

//////////////////配准过程//////////////////////////////////////////////////////////////////
i_p_t align(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_source0, pcl::PointCloud<pcl::PointXYZ>::Ptr key_source0, pcl::PointCloud<pcl::FPFHSignature33>::Ptr features_source0,
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_target, pcl::PointCloud<pcl::PointXYZ>::Ptr key_target, pcl::PointCloud<pcl::FPFHSignature33>::Ptr features_target, float leaf_size, int i = 0) {

	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_source(new pcl::PointCloud<pcl::PointXYZ>);
	*cloud_source = *cloud_source0;
	pcl::PointCloud<pcl::PointXYZ>::Ptr key_source(new pcl::PointCloud<pcl::PointXYZ>);
	*key_source = *key_source0;
	pcl::PointCloud<pcl::FPFHSignature33>::Ptr features_source(new pcl::PointCloud<pcl::FPFHSignature33>);
	*features_source = *features_source0;
	double start = 0, end = 0;
	i_p_t result;
	result.i = i;
	////////////////////初始对应关系估计////////////////////////////////////////////////////////////////////////
	start = GetTickCount();
	pcl::CorrespondencesPtr corr(new pcl::Correspondences());
	float dis = 0.5f;
	*corr = *com_correspondence(cloud_source, key_source, features_source, cloud_target, key_target, features_target, dis, leaf_size);
	if (corr->size() < 3)
		return result;
	pcl::registration::CorrespondenceRejectorOneToOne coo;
	coo.setInputCorrespondences(corr);
	coo.getRemainingCorrespondences(*corr, *corr);
	end = GetTickCount();
	if (corr->size() < 3)
		return result;
	cout << "初始对应关系数目：" << corr->size() << endl;
	cout << "初始对应关系估计：" << end - start << "ms" << endl;
	//show_line(cloud_source, cloud_target, key_source, key_target, corr, leaf_size);
	/////////////////////提取初始对应关系关键点和特征///////////////////////////////////////////////////////////////////////
	pcl::PointCloud<pcl::PointXYZ>::Ptr new_key_source(new pcl::PointCloud<pcl::PointXYZ>());
	pcl::PointCloud<pcl::PointXYZ>::Ptr new_key_target(new pcl::PointCloud<pcl::PointXYZ>());
	pcl::PointCloud<pcl::FPFHSignature33>::Ptr new_features_source(new pcl::PointCloud<pcl::FPFHSignature33>);
	pcl::PointCloud<pcl::FPFHSignature33>::Ptr new_features_target(new pcl::PointCloud<pcl::FPFHSignature33>);
	for (int i = 0; i < corr->size(); i++) {
		new_key_source->push_back(key_source->points[corr->at(i).index_query]);
		new_key_target->push_back(key_target->points[corr->at(i).index_match]);
		new_features_source->push_back(features_source->points[corr->at(i).index_query]);
		new_features_target->push_back(features_target->points[corr->at(i).index_match]);
	}
	key_source = nullptr;
	features_source = nullptr;

	/////////////////去除错误对应关系1///////////////////////////////////////////////////////////////////////////////////////////
	start = GetTickCount();
	*corr = *com_correspondence(cloud_source, new_key_source, new_features_source, cloud_target, new_key_target, new_features_target, dis, leaf_size);
	if (corr->size() < 3)
		return result;
	coo.setInputCorrespondences(corr);
	coo.getRemainingCorrespondences(*corr, *corr);
	end = GetTickCount();
	if (corr->size() < 3)
		return result;
	cout << "对应关系1数目：" << corr->size() << endl;
	cout << "对应关系1估计：" << end - start << "ms" << endl;
	//show_line(cloud_source, cloud_target, new_key_source, new_key_target, corr, leaf_size);
	pcl::PointCloud<pcl::PointXYZ>::Ptr new2_key_source(new pcl::PointCloud<pcl::PointXYZ>());
	pcl::PointCloud<pcl::PointXYZ>::Ptr new2_key_target(new pcl::PointCloud<pcl::PointXYZ>());
	pcl::PointCloud<pcl::FPFHSignature33>::Ptr new2_features_source(new pcl::PointCloud<pcl::FPFHSignature33>);
	pcl::PointCloud<pcl::FPFHSignature33>::Ptr new2_features_target(new pcl::PointCloud<pcl::FPFHSignature33>);
	for (int i = 0; i < corr->size(); i++) {
		new2_key_source->push_back(new_key_source->points[corr->at(i).index_query]);
		new2_key_target->push_back(new_key_target->points[corr->at(i).index_match]);
		new2_features_source->push_back(new_features_source->points[corr->at(i).index_query]);
		new2_features_target->push_back(new_features_target->points[corr->at(i).index_match]);
	}

	new_key_source = nullptr;
	new_key_target = nullptr;
	new_features_source = nullptr;
	new_features_target = nullptr;

	if (corr->size() < 3)
		return result;
	//////////////////////SVD////////////////////////////////////////////////////////////////////////
	if (corr->size() < 10) {
		start = GetTickCount();
		pcl::registration::TransformationEstimationSVD<pcl::PointXYZ, pcl::PointXYZ, float> svd;
		Eigen::Matrix4f trans;
		svd.estimateRigidTransformation(*new2_key_source, *new2_key_target, trans);
		pcl::transformPointCloud(*new2_key_source, *new2_key_source, trans);
		pcl::transformPointCloud(*cloud_source, *cloud_source, trans);


		new2_key_source = nullptr;
		new2_key_target = nullptr;
		new2_features_source = nullptr;
		new2_features_target = nullptr;

		result.trans = trans;
		end = GetTickCount();
		//cout << "SVD：" << end - start << "ms" << endl;
		//pcl::visualization::PCLVisualizer viewer_svd("viewer_svd");
		//viewer_svd.setBackgroundColor(255, 255, 255);
		//viewer_svd.addPointCloud(cloud_target, pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>(cloud_target, 0.0, 255.0, 0.0), "cloud_target");
		//viewer_svd.addPointCloud(cloud_source, pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>(cloud_source, 0.0, 0.0, 255.0), "cloud_source");
		//viewer_svd.spin();

		//////////////////////ICP////////////////////////////////////////////////////////////////////////
		start = GetTickCount();
		pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
		icp.setInputSource(cloud_source);
		icp.setInputTarget(cloud_target);
		icp.setTransformationEpsilon(leaf_size);
		icp.setMaxCorrespondenceDistance(10.0f*leaf_size);
		icp.setMaximumIterations(5000);
		icp.align(*cloud_source);
		end = GetTickCount();
		//cout << "ICP：" << end - start << "ms" << endl;
		//std::cout << " score: " << icp.getFitnessScore(5.0f*leaf_size) << std::endl;
		//std::cout << icp.getFinalTransformation() << std::endl;
		result.trans = result.trans*icp.getFinalTransformation();
		pcl::transformPointCloud(*cloud_source, *cloud_source, icp.getFinalTransformation());
		//pcl::visualization::PCLVisualizer viewer_icp("viewer_icp");
		//viewer_icp.setBackgroundColor(255, 255, 255);
		//viewer_icp.addPointCloud(cloud_source, pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>(cloud_source, 0.0, 255.0, 0.0), "cloud_source");
		//viewer_icp.addPointCloud(cloud_target, pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>(cloud_target, 0.0, 0.0, 255.0), "cloud_target");
		//viewer_icp.spin();
	}
	else {
		////////////////////////随机采样一致性//////////////////////////////////////////////////////////////////////
		start = GetTickCount();
		pcl::SampleConsensusPrerejective<pcl::PointXYZ, pcl::PointXYZ, pcl::FPFHSignature33> align;

		pcl::PointCloud<pcl::PointXYZ>::Ptr temp(new pcl::PointCloud<pcl::PointXYZ>);
		align.setInputSource(new2_key_source);
		align.setSourceFeatures(new2_features_source);
		align.setInputTarget(new2_key_target);
		align.setTargetFeatures(new2_features_target);
		align.setMaximumIterations(300); // Number of RANSAC iterations
		align.setNumberOfSamples(5); // Number of points to sample for generating/prerejecting a pose
		align.setCorrespondenceRandomness(1); // Number of nearest features to use
		align.setSimilarityThreshold(0.9f); // Polygonal edge length similarity threshold
		align.setMaxCorrespondenceDistance(5.0f*leaf_size); // Inlier threshold
		//align.setRANSACOutlierRejectionThreshold(5.0f * leaf_size);
		align.setInlierFraction(0.25f); // Required inlier fraction for accepting a pose hypothesis
		align.align(*temp);

		temp = nullptr;

		vector<int> indice;
		indice = align.getInliers();
		if (indice.size() < 5)
			return result;
		end = GetTickCount();
		//cout << "随机采样一致性：" << end - start << "ms" << endl;
		//cout << "分数： " << align.getFitnessScore(5.0f*leaf_size) << endl;
		cout << "内点数：" << indice.size() << endl;

		pcl::CorrespondencesPtr corr1(new pcl::Correspondences());
		for (int i = 0; i < indice.size(); i++) {
			pcl::Correspondence corri(indice[i], indice[i], 0);
			corr1->push_back(corri);
		}

		//show_line(cloud_source, cloud_target, new2_key_source, new2_key_target, corr1, leaf_size);
		pcl::PointCloud<pcl::PointXYZ>::Ptr new3_key_source(new pcl::PointCloud<pcl::PointXYZ>());
		pcl::PointCloud<pcl::PointXYZ>::Ptr new3_key_target(new pcl::PointCloud<pcl::PointXYZ>());
		pcl::PointCloud<pcl::FPFHSignature33>::Ptr new3_features_source(new pcl::PointCloud<pcl::FPFHSignature33>);
		pcl::PointCloud<pcl::FPFHSignature33>::Ptr new3_features_target(new pcl::PointCloud<pcl::FPFHSignature33>);
		for (int i = 0; i < corr1->size(); i++) {
			new3_key_source->push_back(new2_key_source->points[corr1->at(i).index_query]);
			new3_key_target->push_back(new2_key_target->points[corr1->at(i).index_match]);
			new3_features_source->push_back(new2_features_source->points[corr1->at(i).index_query]);
			new3_features_target->push_back(new2_features_target->points[corr1->at(i).index_match]);
		}
		new2_key_source = nullptr;
		new2_key_target = nullptr;
		new2_features_source = nullptr;
		new2_features_target = nullptr;

		/////////////////去除错误对应关系2///////////////////////////////////////////////////////////////////////////////////////////
		start = GetTickCount();
		*corr1 = *com_correspondence(cloud_source, new3_key_source, new3_features_source, cloud_target, new3_key_target, new3_features_target, dis, leaf_size);
		if (corr1->size() < 3)
			return result;
		coo.setInputCorrespondences(corr1);
		coo.getRemainingCorrespondences(*corr1, *corr1);
		end = GetTickCount();
		if (corr1->size() < 3)
			return result;
		cout << "对应关系2数目：" << corr1->size() << endl;
		cout << "对应关系2估计：" << end - start << "ms" << endl;
		//show_line(cloud_source, cloud_target, new3_key_source, new3_key_target, corr1, leaf_size);
		pcl::PointCloud<pcl::PointXYZ>::Ptr new4_key_source(new pcl::PointCloud<pcl::PointXYZ>());
		pcl::PointCloud<pcl::PointXYZ>::Ptr new4_key_target(new pcl::PointCloud<pcl::PointXYZ>());

		for (int i = 0; i < corr1->size(); i++) {
			new4_key_source->push_back(new3_key_source->points[corr1->at(i).index_query]);
			new4_key_target->push_back(new3_key_target->points[corr1->at(i).index_match]);
		}
		new3_key_source = nullptr;
		new3_key_target = nullptr;

		start = GetTickCount();
		pcl::registration::TransformationEstimationSVD<pcl::PointXYZ, pcl::PointXYZ, float> svd;
		Eigen::Matrix4f trans;
		svd.estimateRigidTransformation(*new4_key_source, *new4_key_target, trans);
		pcl::transformPointCloud(*new4_key_source, *new4_key_source, trans);
		pcl::transformPointCloud(*cloud_source, *cloud_source, trans);

		new4_key_source = nullptr;
		new4_key_target = nullptr;

		result.trans = trans;
		end = GetTickCount();
		//cout << "SVD：" << end - start << "ms" << endl;
		//pcl::visualization::PCLVisualizer viewer_svd("viewer_svd");
		//viewer_svd.setBackgroundColor(255, 255, 255);
		//viewer_svd.addPointCloud(cloud_target, pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>(cloud_target, 0.0, 255.0, 0.0), "cloud_target");
		//viewer_svd.addPointCloud(cloud_source, pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>(cloud_source, 0.0, 0.0, 255.0), "cloud_source");
		//viewer_svd.spin();

		//////////////////////ICP////////////////////////////////////////////////////////////////////////
		start = GetTickCount();
		pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
		icp.setInputSource(cloud_source);
		icp.setInputTarget(cloud_target);
		icp.setTransformationEpsilon(leaf_size);
		icp.setMaxCorrespondenceDistance(10.0f*leaf_size);
		icp.setMaximumIterations(5000);
		icp.align(*cloud_source);
		result.trans = result.trans*icp.getFinalTransformation();
		end = GetTickCount();
		pcl::transformPointCloud(*cloud_source, *cloud_source, icp.getFinalTransformation());
		//cout << "ICP：" << end - start << "ms" << endl;
		//cout << " score: " << icp.getFitnessScore(5.0f*leaf_size) << std::endl;
		//pcl::visualization::PCLVisualizer viewer_icp("viewer_icp");
		//viewer_icp.setBackgroundColor(255, 255, 255);
		//viewer_icp.addPointCloud(cloud_source, pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>(cloud_source, 0.0, 255.0, 0.0), "cloud_source");
		//viewer_icp.addPointCloud(cloud_target, pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>(cloud_target, 0.0, 0.0, 255.0), "cloud_target");
		//viewer_icp.spin();
	}
	result.percent = com_overlap_rate(cloud_source, cloud_target, 5.0f*leaf_size);
	result.trans = result.trans.inverse().eval();
	return result;
}

void full_align(string name_source, string name_target) {

	//////////////////////读取点云//////////////////////////////////////////////
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_source(new pcl::PointCloud<pcl::PointXYZ>);
	//pcl::io::loadPLYFile("D:/code/PCD/自建配准点云/scene+rt/h1.ply", *cloud_source);
	pcl::io::loadPLYFile(name_source, *cloud_source);
	cout << "滤波前源点云点数：" << cloud_source->size() << endl;
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_target(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::io::loadPLYFile(name_target, *cloud_target);
	cout << "滤波前目标点云点数：" << cloud_target->size() << endl;
	//*cloud_source = *add_gaussian_noise(cloud_source, 0.2);
	//*cloud_target = *add_gaussian_noise(cloud_target, 0.2);
	double start = 0;
	double end = 0;
	//////////////////////循环体素滤波//////////////////////////////////////////////////////////////////////
	start = GetTickCount();
	float leaf_size = 0;
	*cloud_source = *point_cloud_to_num(cloud_source, 10000);
	end = GetTickCount();
	leaf_size = com_leaf(cloud_source);
	end = GetTickCount();
	cout << "源点云分辨率：" << leaf_size << endl;
	cout << "源点云点数：" << cloud_source->size() << endl;
	cout << "源点云循环体素滤波：" << end - start << "ms" << endl;
	//pcl::io::savePLYFile("e:/boy1.ply", *cloud_source);

	//////////////////////循环体素滤波//////////////////////////////////////////////////////////////////////
	start = GetTickCount();
	*cloud_target = *voxel_grid(cloud_target, leaf_size);
	*cloud_target = *point_cloud_to_leaf_size(cloud_target, leaf_size);
	leaf_size = com_leaf(cloud_target);
	end = GetTickCount();
	cout << "目标点云分辨率：" << leaf_size << endl;
	cout << "目标点云点数：" << cloud_target->size() << endl;
	cout << "目标点云循环体素滤波：" << end - start << "ms" << endl;
	//pcl::io::savePLYFile("e:/boy2.ply", *cloud_target);
	*cloud_source = *voxel_grid(cloud_source, leaf_size);
	*cloud_target = *voxel_grid(cloud_target, leaf_size);

	/////////////////源点云特征估计///////////////////////////////////////////////////////////////////////////
	leaf_size = com_leaf(cloud_source);
	pcl::PointCloud<pcl::Normal>::Ptr normals_source(new pcl::PointCloud<pcl::Normal>());
	*normals_source = *normal_estimation_OMP(cloud_source, leaf_size*5.0f);
	pcl::PointCloud<pcl::FPFHSignature33>::Ptr features_source(new pcl::PointCloud<pcl::FPFHSignature33>());
	pcl::PointCloud<pcl::PointXYZ>::Ptr key_source(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree_source(new pcl::search::KdTree<pcl::PointXYZ>());
	tree_source->setInputCloud(cloud_source);
	start = GetTickCount();
	*key_source = *key_detect(cloud_source, normals_source, tree_source, 3.0f * leaf_size);
	//*key_source = *key_detect_u(cloud_source, 20.0*leaf_size);//对比实验
	*features_source = *com_features(cloud_source, normals_source, tree_source, key_source, 15.0f * leaf_size);
	//*features_source = *com_pfh_feature(cloud_source, normals_source, key_source, 5.0f *leaf_size);//对比实验
	end = GetTickCount();
	cout << "源点云关键点数目：" << key_source->size() << endl;
	cout << "源点云特征估计：" << end - start << "ms" << endl;
	//show_key_point(cloud_source, key_source);

	//////////////////目标点云特征估计//////////////////////////////////////////////////////////////////////////
	//leaf_size = com_leaf(cloud_target);
	pcl::PointCloud<pcl::Normal>::Ptr normals_target(new pcl::PointCloud<pcl::Normal>());
	*normals_target = *normal_estimation_OMP(cloud_target, leaf_size*5.0f);
	pcl::PointCloud<pcl::FPFHSignature33>::Ptr features_target(new pcl::PointCloud<pcl::FPFHSignature33>());
	pcl::PointCloud<pcl::PointXYZ>::Ptr key_target(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree_target(new pcl::search::KdTree<pcl::PointXYZ>());
	tree_target->setInputCloud(cloud_target);
	start = GetTickCount();
	*key_target = *key_detect(cloud_target, normals_target, tree_target, 3.0f * leaf_size);
	//*key_target = *key_detect_u(cloud_target, 20.0*leaf_size);//对比实验
	*features_target = *com_features(cloud_target, normals_target, tree_target, key_target, 15.0f * leaf_size);
	//*features_target = *com_pfh_feature(cloud_target, normals_target, key_target, 5.0f *leaf_size);//对比实验
	end = GetTickCount();
	cout << "目标点云关键点数目：" << key_target->size() << endl;
	cout << "目标点云特征估计：" << end - start << "ms" << endl;
	//show_key_point(cloud_target, key_target);
	i_p_t result = align(cloud_source, key_source, features_source, cloud_target, key_target, features_target, leaf_size);
	return;
}

void full_align_predict(string name_source, string name_target) {

	//////////////////////读取点云//////////////////////////////////////////////
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_source(new pcl::PointCloud<pcl::PointXYZ>);
	//pcl::io::loadPLYFile("D:/code/PCD/自建配准点云/scene+rt/h1.ply", *cloud_source);
	pcl::io::loadPLYFile(name_source, *cloud_source);
	cout << "滤波前源点云点数：" << cloud_source->size() << endl;
	cout << "源点云分辨率：" << com_leaf(cloud_source) << endl;
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_target(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::io::loadPLYFile(name_target, *cloud_target);
	cout << "滤波前目标点云点数：" << cloud_target->size() << endl;
	cout << "目标点云分辨率：" << com_leaf(cloud_target) << endl;

	//*cloud_source = *add_gaussian_noise(cloud_source, 0.02);
	//*cloud_target = *add_gaussian_noise(cloud_target, 0.2);
	//show_point_cloud(cloud_source);
	//show_point_cloud(cloud_target);
	double start = 0;
	double end = 0;
	float leaf_size = 1;
	/////////////////源点云特征估计///////////////////////////////////////////////////////////////////////////
	//leaf_size = com_leaf(cloud_source);
	pcl::PointCloud<pcl::Normal>::Ptr normals_source(new pcl::PointCloud<pcl::Normal>());
	*normals_source = *normal_estimation_OMP(cloud_source, leaf_size*5.0f);
	pcl::PointCloud<pcl::FPFHSignature33>::Ptr features_source(new pcl::PointCloud<pcl::FPFHSignature33>());
	pcl::PointCloud<pcl::PointXYZ>::Ptr key_source(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree_source(new pcl::search::KdTree<pcl::PointXYZ>());
	tree_source->setInputCloud(cloud_source);
	start = GetTickCount();
	*key_source = *key_detect(cloud_source, normals_source, tree_source, 3.0f * leaf_size);
	//*key_source = *key_detect_u(cloud_source, 20.0f*leaf_size);
	*features_source = *com_features(cloud_source, normals_source, tree_source, key_source, 5.0f * leaf_size);
	normals_source = nullptr;
	tree_source = nullptr;
	end = GetTickCount();
	cout << "源点云关键点数目：" << key_source->size() << endl;
	cout << "源点云特征估计：" << end - start << "ms" << endl;
	//////////////////目标点云特征估计//////////////////////////////////////////////////////////////////////////
	//leaf_size = com_leaf(cloud_target);
	pcl::PointCloud<pcl::Normal>::Ptr normals_target(new pcl::PointCloud<pcl::Normal>());
	*normals_target = *normal_estimation_OMP(cloud_target, leaf_size*5.0f);
	pcl::PointCloud<pcl::FPFHSignature33>::Ptr features_target(new pcl::PointCloud<pcl::FPFHSignature33>());
	pcl::PointCloud<pcl::PointXYZ>::Ptr key_target(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree_target(new pcl::search::KdTree<pcl::PointXYZ>());
	tree_target->setInputCloud(cloud_target);
	start = GetTickCount();
	*key_target = *key_detect(cloud_target, normals_target, tree_target, 3.0f * leaf_size);
	//*key_target = *key_detect_u(cloud_target, 20.0f*leaf_size);
	*features_target = *com_features(cloud_target, normals_target, tree_target, key_target, 5.0f * leaf_size);
	normals_target = nullptr;
	tree_target = nullptr;

	end = GetTickCount();
	cout << "目标点云关键点数目：" << key_target->size() << endl;
	cout << "目标点云特征估计：" << end - start << "ms" << endl;
	//show_key_point(cloud_source, key_source);
	//show_key_point(cloud_target, key_target);
	i_p_t result = align(cloud_source, key_source, features_source, cloud_target, key_target, features_target, leaf_size);
	return;
}

//////////////////对比实验配准过程//////////////////////////////////////////////////////////////////
float align_pfh(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_source0, pcl::PointCloud<pcl::PointXYZ>::Ptr key_source0, pcl::PointCloud<pcl::PFHSignature125>::Ptr features_source0,
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_target, pcl::PointCloud<pcl::PointXYZ>::Ptr key_target, pcl::PointCloud<pcl::PFHSignature125>::Ptr features_target, float leaf_size) {
	double start = 0;
	double end = 0;
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_source(new pcl::PointCloud<pcl::PointXYZ>);
	*cloud_source = *cloud_source0;
	pcl::PointCloud<pcl::PointXYZ>::Ptr key_source(new pcl::PointCloud<pcl::PointXYZ>);
	*key_source = *key_source0;
	pcl::PointCloud<pcl::PFHSignature125>::Ptr features_source(new pcl::PointCloud<pcl::PFHSignature125>);
	*features_source = *features_source0;

	////////////////////初始对应关系估计////////////////////////////////////////////////////////////////////////
	start = GetTickCount();
	pcl::CorrespondencesPtr corr(new pcl::Correspondences());
	float dis = 20;
	*corr = *com_correspondence(features_source, features_target, dis);
	end = GetTickCount();
	cout << "初始对应关系数目：" << corr->size() << endl;
	cout << "初始对应关系估计：" << end - start << "ms" << endl;
	if (corr->size() < 5)
		return 0;
	//show_coor(cloud_source, cloud_target, *key_source, *key_target, features_source, features_target, corr);
	show_line(cloud_source, cloud_target, key_source, key_target, corr, leaf_size);
	/////////////////////提取初始对应关系关键点和特征///////////////////////////////////////////////////////////////////////
	pcl::PointCloud<pcl::PointXYZ>::Ptr new_key_source(new pcl::PointCloud<pcl::PointXYZ>());
	pcl::PointCloud<pcl::PointXYZ>::Ptr new_key_target(new pcl::PointCloud<pcl::PointXYZ>());
	pcl::PointCloud<pcl::PFHSignature125>::Ptr new_features_source(new pcl::PointCloud<pcl::PFHSignature125>);
	pcl::PointCloud<pcl::PFHSignature125>::Ptr new_features_target(new pcl::PointCloud<pcl::PFHSignature125>);
	for (int i = 0; i < corr->size(); i++) {
		new_key_source->push_back(key_source->points[corr->at(i).index_query]);
		new_key_target->push_back(key_target->points[corr->at(i).index_match]);
		new_features_source->push_back(features_source->points[corr->at(i).index_query]);
		new_features_target->push_back(features_target->points[corr->at(i).index_match]);
	}

	////////////////////////随机采样一致性//////////////////////////////////////////////////////////////////////
	start = GetTickCount();
	pcl::SampleConsensusPrerejective<pcl::PointXYZ, pcl::PointXYZ, pcl::PFHSignature125> align;
	align.setInputSource(new_key_source);
	align.setSourceFeatures(new_features_source);
	align.setInputTarget(new_key_target);
	align.setTargetFeatures(new_features_target);
	align.setMaximumIterations(5000); // Number of RANSAC iterations
	align.setNumberOfSamples(5); // Number of points to sample for generating/prerejecting a pose
	align.setCorrespondenceRandomness(1); // Number of nearest features to use
	align.setSimilarityThreshold(0.9f); // Polygonal edge length similarity threshold
	align.setMaxCorrespondenceDistance(10.0f*leaf_size); // Inlier threshold
	align.setInlierFraction(0.25f); // Required inlier fraction for accepting a pose hypothesis
	align.align(*new_key_source);
	end = GetTickCount();
	//cout << "随机采样一致性：" << end - start << "ms" << endl;
	//cout << "分数： " << align.getFitnessScore(leaf_size) << endl;
	Eigen::Matrix4f transformation = align.getFinalTransformation();
	pcl::console::print_info("Inliers: %i/%i\n", align.getInliers().size(), new_key_source->size());
	pcl::transformPointCloud(*cloud_source, *cloud_source, transformation);
	if (align.getInliers().size() < 5)
		return 0;
	// Show alignment
	pcl::visualization::PCLVisualizer visu1("Alignment1");
	visu1.setBackgroundColor(255, 255, 255);
	visu1.addPointCloud(cloud_source, pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>(cloud_source, 0.0, 255.0, 0.0), "scene1");
	visu1.addPointCloud(cloud_target, pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>(cloud_target, 0.0, 0.0, 255.0), "object_aligned1");
	visu1.spin();

	*cloud_source = *voxel_grid(cloud_source, 5.0f*leaf_size);
	////////////////////ICP////////////////////////////////////////////////////////////////////////
	start = GetTickCount();
	pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
	icp.setInputSource(cloud_source);
	icp.setInputTarget(cloud_target);
	icp.setTransformationEpsilon(leaf_size);
	icp.setMaxCorrespondenceDistance(10.0f * leaf_size);
	icp.setMaximumIterations(3000);
	icp.align(*cloud_source);
	end = GetTickCount();
	//cout << "ICP：" << end - start << "ms" << endl;
	//std::cout << "has converged:" << icp.hasConverged() << " score: " << icp.getFitnessScore(5.0f*leaf_size) << std::endl;
	//std::cout << icp.getFinalTransformation() << std::endl;
	pcl::transformPointCloud(*cloud_source, *cloud_source, icp.getFinalTransformation());
	//pcl::visualization::PCLVisualizer visu("Alignment");
	//visu.setBackgroundColor(255, 255, 255);
	//visu.addPointCloud(cloud_source, pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>(cloud_source, 0.0, 255.0, 0.0), "scene1");
	//visu.addPointCloud(cloud_target, pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>(cloud_target, 0.0, 0.0, 255.0), "object_aligned1");
	//visu.spin();

	float percent = com_overlap_rate(cloud_source, cloud_target, 5.0f*leaf_size);

	if (percent > 0.8f)
		return percent;
	return 0;
}

void full_align_pfh(string name_source, string name_target) {

	//////////////////////读取点云//////////////////////////////////////////////
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_source(new pcl::PointCloud<pcl::PointXYZ>);
	//pcl::io::loadPLYFile("D:/code/PCD/自建配准点云/scene+rt/h1.ply", *cloud_source);
	pcl::io::loadPLYFile(name_source, *cloud_source);
	cout << "滤波前源点云点数：" << cloud_source->size() << endl;
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_target(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::io::loadPLYFile(name_target, *cloud_target);
	cout << "滤波前目标点云点数：" << cloud_target->size() << endl;
	//*cloud_source = *add_gaussian_noise(cloud_source, 0.2);
	//*cloud_target = *add_gaussian_noise(cloud_target, 0.2);
	double start = 0;
	double end = 0;
	//////////////////////循环体素滤波//////////////////////////////////////////////////////////////////////
	start = GetTickCount();
	float leaf_size = 0;
	*cloud_source = *point_cloud_to_num(cloud_source, 10000);
	end = GetTickCount();
	leaf_size = com_leaf(cloud_source);
	end = GetTickCount();
	cout << "源点云分辨率：" << leaf_size << endl;
	cout << "源点云点数：" << cloud_source->size() << endl;
	cout << "源点云循环体素滤波：" << end - start << "ms" << endl;
	//pcl::io::savePLYFile("e:/boy1.ply", *cloud_source);

	//////////////////////循环体素滤波//////////////////////////////////////////////////////////////////////
	start = GetTickCount();
	*cloud_target = *voxel_grid(cloud_target, leaf_size);
	*cloud_target = *point_cloud_to_leaf_size(cloud_target, leaf_size);
	leaf_size = com_leaf(cloud_target);
	end = GetTickCount();
	cout << "目标点云分辨率：" << leaf_size << endl;
	cout << "目标点云点数：" << cloud_target->size() << endl;
	cout << "目标点云循环体素滤波：" << end - start << "ms" << endl;
	//pcl::io::savePLYFile("e:/boy2.ply", *cloud_target);
	*cloud_source = *voxel_grid(cloud_source, leaf_size);
	*cloud_target = *voxel_grid(cloud_target, leaf_size);

	/////////////////源点云特征估计///////////////////////////////////////////////////////////////////////////
	leaf_size = com_leaf(cloud_source);
	pcl::PointCloud<pcl::Normal>::Ptr normals_source(new pcl::PointCloud<pcl::Normal>());
	*normals_source = *normal_estimation_OMP(cloud_source, leaf_size*5.0f);
	pcl::PointCloud<pcl::PFHSignature125>::Ptr features_source(new pcl::PointCloud<pcl::PFHSignature125>());
	pcl::PointCloud<pcl::PointXYZ>::Ptr key_source(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree_source(new pcl::search::KdTree<pcl::PointXYZ>());
	tree_source->setInputCloud(cloud_source);
	start = GetTickCount();
	*key_source = *key_detect_u(cloud_source, 20.0*leaf_size);//对比实验
	*features_source = *com_pfh_feature(cloud_source, normals_source, key_source, 5.0f *leaf_size);//对比实验
	end = GetTickCount();
	cout << "源点云关键点数目：" << key_source->size() << endl;
	cout << "源点云特征估计：" << end - start << "ms" << endl;
	//show_key_point(cloud_source, key_source);

	//////////////////目标点云特征估计//////////////////////////////////////////////////////////////////////////
	leaf_size = com_leaf(cloud_target);
	pcl::PointCloud<pcl::Normal>::Ptr normals_target(new pcl::PointCloud<pcl::Normal>());
	*normals_target = *normal_estimation_OMP(cloud_target, leaf_size*5.0f);
	pcl::PointCloud<pcl::PFHSignature125>::Ptr features_target(new pcl::PointCloud<pcl::PFHSignature125>());
	pcl::PointCloud<pcl::PointXYZ>::Ptr key_target(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree_target(new pcl::search::KdTree<pcl::PointXYZ>());
	tree_target->setInputCloud(cloud_target);
	start = GetTickCount();
	*key_target = *key_detect_u(cloud_target, 20.0*leaf_size);//对比实验
	*features_target = *com_pfh_feature(cloud_target, normals_target, key_target, 5.0f *leaf_size);//对比实验
	end = GetTickCount();
	cout << "目标点云关键点数目：" << key_target->size() << endl;
	cout << "目标点云特征估计：" << end - start << "ms" << endl;
	//show_key_point(cloud_target, key_target);
	float percent = align_pfh(cloud_source, key_source, features_source, cloud_target, key_target, features_target, leaf_size);
	return;
}

void align_omp(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_source, pcl::PointCloud<pcl::PointXYZ>::Ptr key_source,
	pcl::PointCloud<pcl::FPFHSignature33>::Ptr features_source, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_target, pcl::PointCloud<pcl::PointXYZ>::Ptr key_target,
	pcl::PointCloud<pcl::FPFHSignature33>::Ptr features_target, float leaf_size, int i) {
	i_p_t result = align(cloud_source, key_source, features_source, cloud_target, key_target, features_target, leaf_size, i);
	m.lock();
	results.push_back(result);
	m.unlock();
	return;
}

bool cmp(i_p_t a, i_p_t b) {
	return a.percent > b.percent;
}

i_p_t predict(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_source, pcl::PointCloud<pcl::PointXYZ>::Ptr key_source, pcl::PointCloud<pcl::FPFHSignature33>::Ptr features_source,
	vector<string> names,
	vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> models_ptr, vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> keys_ptr, vector<pcl::PointCloud<pcl::FPFHSignature33>::Ptr> features_ptr) {

	double start = 0;
	double end = 0;
	float leaf_size = 1;
	results.clear();

	start = GetTickCount();
	for (int i = 0; i < names.size(); i++) {
		thread t(align_omp, cloud_source, key_source, features_source, models_ptr[i], keys_ptr[i], features_ptr[i], leaf_size, i);
		t.detach();
	}
	while (results.size() != names.size()) {
		Sleep(100);
	}
	end = GetTickCount();
	cout << "识别时间：" << end - start << "ms" << endl;
	sort(results.begin(), results.end(), cmp);
	return results[0];
}

void com_key_feature(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_source,float leaf_size=1) {
	pcl::PointCloud<pcl::Normal>::Ptr normals_source(new pcl::PointCloud<pcl::Normal>());
	*normals_source = *normal_estimation_OMP(cloud_source, leaf_size*5.0f);
	pcl::PointCloud<pcl::PointXYZ>::Ptr key_source(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::FPFHSignature33>::Ptr features_source(new pcl::PointCloud<pcl::FPFHSignature33>());
	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree_source(new pcl::search::KdTree<pcl::PointXYZ>());
	tree_source->setInputCloud(cloud_source);
	double start = 0;
	double end = 0;
	start = GetTickCount();
	*key_source = *key_detect(cloud_source, normals_source, tree_source, 3.0f * leaf_size);
	*features_source = *com_features(cloud_source, normals_source, tree_source, key_source, 5.0f * leaf_size);
	end = GetTickCount();
	//cout << "源点云关键点数目：" << key_source->size() << endl;
	//cout << "源点云特征估计：" << end - start << "ms" << endl;
	c_k_f cloud_key_feature;
	cloud_key_feature.cloud = cloud_source;
	cloud_key_feature.key = key_source;
	cloud_key_feature.features = features_source;
	m.lock();
	clouds_keys_features.push_back(cloud_key_feature);
	m.unlock();
	return;
}

//////////配准调试/////////////////////////////////////////
//int main(int argc, char** argv) {
//	vector<string> names = { "armadillo", "bunny", "cat","centaur","cheff", "chicken","david","dog", "dragon","face",
//	"ganesha","gorilla","horse","para" ,"trex","wolf" };
//	//vector<string> names = { "a0","a1", "a2","b0","b1","b2","cat0","cat1","cat2","cen0","cen1","cen2","cheff0","cheff1","cheff2","chicken0","chicken1","chicken2",
//	//"d0","d1","d2","dog0","dog1","dog2","gan0","gan1","gan2","para0","para1","para2","trex0","trex1","trex2" };
//	for (int i = 0; i < names.size(); i++) {
//		cout << names[i] << "/////////////////////////////////////////////////////"<<endl;
//		string name_source = "D:/code/PCD/识别点云/scene/filter/"+names[i]+".ply";
//		string name_target = "D:/code/PCD/识别点云/model/filter/"+names[i]+".ply";
//		full_align_predict(name_source, name_target);
//		cout << "/////////////////////////////////////////////////////" << endl;
//	}
//	//string name_source = "D:/code/PCD/自建配准点云2/dog0.ply";
//	//string name_target = "D:/code/PCD/自建配准点云2/dog1.ply";
//	//full_align(name_source, name_target);
//	return 0;
//}

/////////////////////filter/////////////////////////////
//int main(int argc, char** argv){
//	vector<string> names = { "armadillo", "bunny", "cat","centaur","cheff", "chicken","david","dog", "dragon","face",
//		"ganesha","gorilla","gun","horse","lioness","para" ,"trex","victoria","wolf" };
//	for (int i = 0; i < names.size(); i++) {
//		pcl::PointCloud<pcl::PointXYZ>::Ptr scene(new pcl::PointCloud<pcl::PointXYZ>);
//		pcl::io::loadPLYFile("D:/code/PCD/识别点云/scene/scene/" + names[i] + ".ply", *scene);
//		*scene = *voxel_grid(scene, 2.0f);
//		pcl::io::savePLYFile("D:/code/PCD/识别点云/scene/filter/" + names[i] + ".ply", *scene);
//
//		pcl::PointCloud<pcl::PointXYZ>::Ptr model(new pcl::PointCloud<pcl::PointXYZ>);
//		pcl::io::loadPLYFile("D:/code/PCD/识别点云/model/model/" + names[i] + ".ply", *model);
//		*model = *voxel_grid(model, 2.0f);
//		pcl::io::savePLYFile("D:/code/PCD/识别点云/model/filter/" + names[i] + ".ply", *model);
//	}
//	return 0;
//}

//int main(int argc, char** argv){
//	string road = "D:/code/PCD/识别点云/model/";
//	vector<string> names = { "armadillo", "bunny", "cat","centaur","cheff", "chicken","david","dog", "dragon","face",
//	"ganesha","gorilla","horse","para" ,"trex","wolf" };
//
//	for (int i = 0; i < names.size(); i++) {
//		string name = "D:/code/PCD/识别点云/model/filter/" + names[i] + ".ply";
//		pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
//		pcl::io::loadPLYFile(name, *cloud);
//		float leaf_size =1;
//
//		pcl::PointCloud<pcl::Normal>::Ptr normal(new pcl::PointCloud<pcl::Normal>());
//		normal = normal_estimation_OMP(cloud, leaf_size*5.0f);
//		pcl::PointCloud<pcl::FPFHSignature33>::Ptr feature(new pcl::PointCloud<pcl::FPFHSignature33>());
//		pcl::PointCloud<pcl::PointXYZ>::Ptr key(new pcl::PointCloud<pcl::PointXYZ>());
//		pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
//		tree->setInputCloud(cloud);
//		*key=*key_detect(cloud, normal, tree, 3.0f * leaf_size);
//		cout <<names[i]<<": "<< key->size() << endl;
//		*feature = *com_features(cloud, normal, tree, key, 5.0f * leaf_size);
//		pcl::io::savePLYFile("D:/code/PCD/识别点云/model/feature/" + names[i] + "_key.ply", *key);
//		pcl::io::savePLYFile( "D:/code/PCD/识别点云/model/feature/" + names[i] + "_feature.ply", *feature);
//	}
//	return 0;
//}

int main(int argc, char** argv) {
	vector<string> names = { "armadillo", "bunny", "cat","centaur","cheff", "chicken","david","dog", "dragon","face",
	"ganesha","gorilla","horse","para" ,"trex","wolf" };
	//vector<string> names = { "armadillo", "bunny","cheff"};
	vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> models;
	vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> keys;
	vector<pcl::PointCloud<pcl::FPFHSignature33>::Ptr> features;
	for (int i = 0; i < names.size(); i++) {
		pcl::PointCloud<pcl::PointXYZ>::Ptr model(new pcl::PointCloud<pcl::PointXYZ>);
		pcl::PointCloud<pcl::PointXYZ>::Ptr key(new pcl::PointCloud<pcl::PointXYZ>);
		pcl::PointCloud<pcl::FPFHSignature33>::Ptr feature(new pcl::PointCloud<pcl::FPFHSignature33>);
		pcl::io::loadPLYFile("D:/code/PCD/识别点云/model/filter/" + names[i] + ".ply", *model);
		pcl::io::loadPLYFile("D:/code/PCD/识别点云/model/feature/" + names[i] + "_key.ply", *key);
		pcl::io::loadPLYFile("D:/code/PCD/识别点云/model/feature/" + names[i] + "_feature.ply", *feature);
		models.push_back(model);
		keys.push_back(key);
		features.push_back(feature);
	}
	pcl::PointCloud<pcl::PointXYZ>::Ptr scenes(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::io::loadPLYFile("D:/code/PCD/识别点云/scene/scenes_filter/bunny_armadillo_cheff.ply", *scenes);
	//pcl::io::loadPLYFile("D:/code/PCD/识别点云/scene/filter/armadillo.ply", *scenes);
	vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> clouds;
	clouds = euclidean_cluster(scenes, 1.5f*10.0f);
	vector<i_p_t> result_final;
	for (int i = 0; i < clouds.size(); i++) {
		thread t(com_key_feature, clouds[i],1.0f);
		t.detach();
	}
	while (clouds_keys_features.size() != clouds.size()) {
		Sleep(100);
	}
	for (int i = 0; i < clouds.size(); i++) {
		result_final.push_back(predict(clouds_keys_features[i].cloud, clouds_keys_features[i].key, clouds_keys_features[i].features, names, models, keys, features));
		if (result_final[i].percent > 0.8)
			cout << names[result_final[i].i] << endl;
		else
			cout << "nnnnnnnnnn" << endl;
	}
	show_point_clouds(clouds);
	show_point_clouds_and_trans_models(clouds, models, result_final);
	return 0;
}