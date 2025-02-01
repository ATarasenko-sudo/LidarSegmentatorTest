#include <iostream>
#include <fstream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/search/kdtree.h>
#include <pcl/filters/passthrough.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>



// Примечание - Данные алгоритмы выбраны исходя из исходных и уже реализованных сегментаторов в PCL библиотеке
//TODO Доделать DBSCAN
//Алгоритм, которого тут нет, однако бы подошел , ввиду того, что тоже очень устойчив к выбросам и шумам
//Также его особенностью является возможность не задвать верхнюю границу для для формирования кластеров. 
int main()
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);

    if (pcl::io::loadPCDFile<pcl::PointXYZ>("../data/test.pcd", *cloud) == -1) //* load the file
    {
        PCL_ERROR("Couldn't read file test.pcd \n");
        return (-1);
    }

    std::cout << "Loaded " << cloud->width * cloud->height
              << " data points from test.pcd with the following fields: " << std::endl;



    // --- 0. Удаление диапазона точек по оси Z. Самый простой способ,однако не подходит для рельефа 
    //Также не подходит, если надо учитывать объекты, располагающиеся близко к земле
    //Однако очень хорошо подходит для выделения ROI 
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PassThrough<pcl::PointXYZ> pass;
    pass.setInputCloud(cloud);
    pass.setFilterFieldName("z");
    pass.setFilterLimits(-2, -1.0); // Диапазон Z: оставляем точки от -1 до 10 метров
    pass.filter(*cloud_filtered);


    // --- 1. RANSAC для выделения плоскости земли ---

    pcl::SACSegmentation<pcl::PointXYZ> seg;
    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);

    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setDistanceThreshold(0.1);
    seg.setMaxIterations(500);
    seg.setInputCloud(cloud_filtered);
    seg.segment(*inliers, *coefficients);

    if (inliers->indices.empty())
    {
        std::cerr << "Ошибка: RANSAC не нашел плоскость" << std::endl;
        return (-1);
    }

    // Выделяем точки земли по RANSAC

    // Один из вариантов сегментации RANSAC. Алгоритм хорош в условиях зашумленности, однако данные требуют предварительной обработки. Также алгоритм хорошо справляется с рельефом дороги
    // (Подъем/спуск/ бордрюры и тд) 
    // в рамках сегментации земли RANSAC находит плоскость, которую будет считать за землю

    // ввиду малых параметров для настройки, лучше приводить обработку данных для выбора диапазона облака точек, ввиду того, что алгоритм может сегментировать здание или другой большой объект

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_ransac_ground(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::ExtractIndices<pcl::PointXYZ> extract;
    extract.setInputCloud(cloud_filtered);
    extract.setIndices(inliers);
    extract.setNegative(false);
    extract.filter(*cloud_ransac_ground);


    // Записываем RANSAC-землю в CSV (использовалд для дебага см. ipynb файл)
    std::ofstream file_ransac("../result/ransac_ground.csv");
    if (!file_ransac.is_open())
    {
        std::cerr << "Ошибка: не удалось открыть файл ransac_ground.csv" << std::endl;
        return (-1);
    }
    file_ransac << "X,Y,Z\n";
    for (const auto &point : cloud_ransac_ground->points)
    {
        file_ransac << point.x << "," << point.y << "," << point.z << "\n";
    }
    file_ransac.close();


    // Удаляем RANSAC-землю из облака
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered_ransac(new pcl::PointCloud<pcl::PointXYZ>);
    extract.setNegative(true);
    extract.filter(*cloud_filtered_ransac);

    std::cout << "Осталось " << cloud_filtered_ransac->size() << " точек после удаления RANSAC-земли." << std::endl;

    // Сохраняем очищенное облако в PCD-файл
    pcl::io::savePCDFileASCII("../result/filtered_ransac.pcd", *cloud_filtered_ransac);

    // --- 2. Евклидова кластеризация для выделения земли ---

    //Самый очевидный вариант использовать евклидову кластеризацию, поскольку подход простой. Мы находим самую большую цепочку из кластеров и принимаем ее за землю,
    //Однако тоже стоит учитывать, что алгоритм уже уязвим к выбросам и работает немного медленее, ввиду чего требуется четкое выделение ROI- region of interest. 
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
    tree->setInputCloud(cloud);

    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
    ec.setClusterTolerance(0.15);
    ec.setMinClusterSize(1000);
    ec.setMaxClusterSize(400000);
    ec.setSearchMethod(tree);
    ec.setInputCloud(cloud);
    ec.extract(cluster_indices);

    if (cluster_indices.empty())
    {
        std::cerr << "Ошибка: кластеризация не нашла ни одного кластера!" << std::endl;
        return (-1);
    }

    // Выбираем самый большой кластер как землю
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_clustered_ground(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointIndices::Ptr clustered_ground_indices(new pcl::PointIndices);

    int largest_cluster_size = 0;
    for (const auto &indices : cluster_indices)
    {
        if (indices.indices.size() > largest_cluster_size)
        {
            largest_cluster_size = indices.indices.size();
            cloud_clustered_ground->clear();
            clustered_ground_indices->indices = indices.indices;
            for (const auto &idx : indices.indices)
            {
                cloud_clustered_ground->points.push_back(cloud->points[idx]);
            }
        }
    }



    // Записываем кластеризированную землю в CSV - тоже для дебага. В ipynb видно, какой участок вырезался из облака точек.
    std::ofstream file_clustered("../result/clustered_ground.csv");
    if (!file_clustered.is_open())
    {
        std::cerr << "Ошибка: не удалось открыть файл clustered_ground.csv" << std::endl;
        return (-1);
    }
    file_clustered << "X,Y,Z\n";
    for (const auto &point : cloud_clustered_ground->points)
    {
        file_clustered << point.x << "," << point.y << "," << point.z << "\n";
    }
    file_clustered.close();

    // Удаляем кластеризированную землю из облака
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered_clustered(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::ExtractIndices<pcl::PointXYZ> extract_clustered;
    extract_clustered.setInputCloud(cloud);
    extract_clustered.setIndices(clustered_ground_indices);
    extract_clustered.setNegative(true);
    extract_clustered.filter(*cloud_filtered_clustered);

    std::cout << "Осталось " << cloud_filtered_clustered->size() << " точек после удаления кластеризированной земли." << std::endl;

    // Сохраняем очищенное облако в PCD-файл
    pcl::io::savePCDFileASCII("../result/filtered_clustered.pcd", *cloud_filtered_clustered);


    return (0);
}
