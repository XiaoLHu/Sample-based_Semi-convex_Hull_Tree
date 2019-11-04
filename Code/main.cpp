/*
*
* Copyright (C) 2017-  Yewang Chen<ywchen@hqu.edu.cn;nalandoo@gmail.com>
* License: GPL v1
* This software may be modified and distributed under the terms
* of license.
*
*/

#include <windows.h>
#include <time.h>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include "basic_functions.h"
#include "ConvexNode.h"
#include "Array.h"
#include "Convex_Tree.h"
#include "cyw_types.h"
#include "CPU_Convex_Tree.h"
#include "CUDA_Convex_Tree.h"
#include "data_processor.h"


typedef struct Convex_Node {
	bool isLeaf;
	int  node_index;
	int  parent_index;
	int  leaf_index;       //the leaf index of this node in all leaf nodes
	int  left_node;
	int  right_node;
} SIMPLE_CONVEX_TREE;


extern "C" int main_vec_add();

extern "C" void call_do_find_approximate_nodes(
	int          candidate_query_points_num,
	FLOAT_TYPE  *candidate_query_points_set,
	int          tree_nodes_num,
	CONVEX_TREE *tree_struct,
	FLOAT_TYPE  *nodes_centers,
	int         *appr_leaf_node_indexes,
	int          DIM
	);

extern "C" int call_cuda_kernel_brute_force(
	int          candidate_query_points_num,
	int         *candidate_query_points_indexes,
	FLOAT_TYPE         *candidate_query_points_set,
	int          data_set_size,
	FLOAT_TYPE         *data_set,
	FLOAT_TYPE         *dist_k_mins_global_tmp,
	int         *idx_k_mins_global_tmp,
	int          K_NN,
	int          DIM);

extern "C" int call_cuda_kernel_brute_force_for_outliers(
	int                candidate_query_points_num,
	int                *candidate_query_points_indexes,
	FLOAT_TYPE         *candidate_query_points_set,
	int                data_set_size,
	FLOAT_TYPE         *data_set,
	int                *data_indexes,
	FLOAT_TYPE         *dist_k_mins_global_tmp,
	int                *idx_k_mins_global_tmp,
	int                K_NN,
	int                DIM);

extern "C" int call_cuda_kernel(
	int          candidate_query_points_num,
	int         *candidate_query_points_indexes,
	FLOAT_TYPE         *candidate_query_points_set,
	int         *candidate_query_points_appr_leaf_node_indexes,
	int          sorted_data_len,
	FLOAT_TYPE         *all_sorted_data_set,
	int         *sorted_data_set_indexes,
	int          tree_nodes_num,
	CONVEX_TREE         *tree_struct,
	int          all_leaf_nodes_constraint_num,
	FLOAT_TYPE         *all_leaf_nodes_ALPHA_set,
	FLOAT_TYPE         *all_leaf_nodes_BETA_set,
	int         *all_constrains_num_of_each_leaf_nodes,
	int         *all_leaf_nodes_offsets_in_all_ALPHA,
	int          leaf_node_num,
	int         *all_leaf_nodes_ancestor_nodes_ids,
	int         *leaf_nodes_start_pos_in_sorted_data_set,
	int         *pts_num_in_sorted_leaf_nodes,
	FLOAT_TYPE         *dist_k_mins_global_tmp,
	int         *idx_k_mins_global_tmp,
	int          K_NN,
	long         *dist_computation_times_arr,
	long         *quadprog_times_arr,
	long         *dist_computation_times_in_quadprog,
	int          NODES_NUM,
	int          DIM
	);

extern "C" int  new_call_cuda_kernel(
	int                 candidate_query_points_num,
	int					*candidate_query_points_indexes,
	FLOAT_TYPE          *candidate_query_points_set,
	int					sorted_data_len,
	FLOAT_TYPE          *all_sorted_data_set,
	int					*sorted_data_set_indexes,
	int					tree_nodes_num,
	CONVEX_TREE         *tree_struct,
	int					all_leaf_nodes_constraint_num,
	FLOAT_TYPE          *all_leaf_nodes_ALPHA_set,
	FLOAT_TYPE          *all_leaf_nodes_BETA_set,
	int					*all_constrains_num_of_each_leaf_nodes,
	int					*all_leaf_nodes_offsets_in_all_ALPHA,
	int					leaf_node_num,
	int					*all_leaf_nodes_ancestor_nodes_ids,
	int					*leaf_nodes_start_pos_in_sorted_data_set,
	int					*pts_num_in_sorted_leaf_nodes,
	FLOAT_TYPE          *dist_k_mins_global_tmp,
	int					*idx_k_mins_global_tmp,
	int					*remain_index,
	int					K_NN,
	long				*dist_computation_times_arr,
	long				*quadprog_times_arr,
	long				*dist_computation_times_in_quadprog,
	int					NODES_NUM,
	int					DIM);


//gof. design patterns: class helper
class Tree_Helper{
public:
	//alg_type specifies the this->ALG_TYPE
	template<typename T>
	static Convex_Tree<T>* create_a_tree(int alg_type, Matrix<T>& data, FLOAT_TYPE leaf_pts_percent){
		Convex_Tree<T>* result = NULL;
		
		if (alg_type == USE_CPU_RECURSIVE_APPRXIMATE_QP || alg_type == USE_CPU_LEAF_APPRXIMATE_QP)
			result = new CPU_Convex_Tree<T>(data, leaf_pts_percent, alg_type);

		if (alg_type == USE_GPU_RECURSIVE_APPRXIMATE_QP || alg_type == USE_GPU_LEAF_APPRXIMATE_QP)
			result = new CUDA_Convex_Tree<T>(data, leaf_pts_percent, alg_type);

		if (result != NULL)	result->set_alg_type(alg_type);
		return result;
	}
};


void Find_min_and_max(Matrix<FLOAT_TYPE> data, float *min, float *max){
	float min_tmp, max_tmp;
	for (int i = 0; i < data.ncols(); i++){
		min_tmp = data[0][i];
		max_tmp = data[0][i];
		for (int j = 0; j < data.nrows(); j++){
			if (min_tmp > data[j][i])
				min_tmp = data[j][i];
			if (max_tmp < data[j][i])
				max_tmp = data[j][i];
		}
		min[i] = min_tmp;
		max[i] = max_tmp;
	}
}


void get_random_data(char *new_data_file_name, char * data_file_name, int query_num, FLOAT_TYPE leaf_percent){
	int dim, data_size;
	float *raw_data = read_data(data_file_name, " ", &dim, &data_size);
	Matrix<FLOAT_TYPE> data(raw_data, data_size, dim);

	float* max = (float*)malloc(dim*sizeof(float));
	float* min = (float*)malloc(dim*sizeof(float));

	Find_min_and_max(data, min, max);

	Matrix<FLOAT_TYPE> query_points = Matrix<FLOAT_TYPE>::new_rand_matrix(query_num, dim, min, max);

	float* query_data = query_points.get_matrix_raw_data();

	Save_data(query_data, dim, query_num, new_data_file_name);
}


void get_raw_data_with_noise(char *new_data_file_name, char * data_file_name, int new_data_size, int noise_rate){
	int dim, data_size;
	float *raw_data = read_data(data_file_name, " ", &dim, &data_size);
	Matrix<FLOAT_TYPE> data(raw_data, data_size, dim);

	Vector<int> new_data_index(new_data_size);
	int level = data_size / new_data_size;
	for (int i = 0; i < new_data_size; i++){
		int index = rand() % level;
		new_data_index[i] = index + i * level;
	}
	Matrix<FLOAT_TYPE> new_data = data.extractRows(new_data_index);
	
	for (int i = 0; i < new_data.nrows(); i++){
		for (int j = 0; j < new_data.ncols(); j++){
			FLOAT_TYPE noise = rand() % noise_rate;
			new_data[i][j] += noise;
		}
	}
	cout << "ending...." << endl;
	float* query_data = new_data.get_matrix_raw_data();
	Save_data(query_data, dim, new_data_size, new_data_file_name);
}


Vector<NNResult<FLOAT_TYPE>* > brute_force_check(Matrix<FLOAT_TYPE>& data,Matrix<FLOAT_TYPE>& q_points,int K,Convex_Tree<FLOAT_TYPE>* cct){
	std::cout << "\n brute force is starting....";
	int dim = data.ncols();
	long start_t = clock();//GetTickCount();
	Vector<NNResult<FLOAT_TYPE>* > results;
	results.resize(q_points.nrows());

	for (int i = 0; i<q_points.nrows(); i++){
		FLOAT_TYPE *q = q_points.get_matrix_raw_data() + i*dim;
		Vector<FLOAT_TYPE> tmp_dists_square = pdist2_squre(data, q, dim);
		results[i] = new NNResult<FLOAT_TYPE>();
		results[i]->init(K);
		for (int j = 0; j<tmp_dists_square.size(); j++){
			results[i]->update(tmp_dists_square[j], j);
		}

		bool err = false;
		for (int j = 0; j<K; j++){
			FLOAT_TYPE cct_result = cct->get_kNN_dists_squre(i)[j];
			FLOAT_TYPE brute_result = results[i]->k_dists_squre[j];
			int cct_result_index = cct->get_kNN_indexes(i)[j];
			int brute_result_index = results[i]->k_indexes[j];
			if ((abs(cct_result - brute_result)>0.001) &&
				(cct_result_index != brute_result_index)){
				std::cout << "\n error found, at point index=" << i << " and j=" << j;
				err = true;
			}
		}
		if (err){
			FLOAT_TYPE* tmp = cct->get_kNN_dists_squre(i);
			int* tmp_indexes = cct->get_kNN_indexes(i);
			std::cout << " \n knn dist squre of query point " << i << " is: ";
			for (i = 0; i<K; i++) {
				std::cout << tmp[i] << ", ";
			}

			std::cout << " \n knn indexes of query point " << i << " is: ";
			for (i = 0; i<K; i++) {
				std::cout << tmp_indexes[i] << ", ";
			}
			results[i]->print();
		}

		//std::cout<<"\n i="<<i<<" ";
		//results[i]->print();
	}
	long end_t = clock();//GetTickCount();
	std::cout << "\n brute force finished, running time is " << end_t - start_t << "s" << endl;
	return results;
}


void test_gpu_brute_force_kNN_on_real_data(int K, char *data_file_name, int test_num_each_batch, int batch_num, FLOAT_TYPE leaf_percent, FILE* log_file){
	int dim;
	int data_size;
	float *raw_data = read_data(data_file_name, " ", &dim, &data_size);
	write_file_log("\n\n", log_file);
	write_file_log(data_file_name, log_file);

	//data_size=40000;
	//---create data set for query
	Matrix<FLOAT_TYPE> data(raw_data, data_size, dim);
	//data /= 1000;
	//---data.print();


	//create a convex tree
	//Convex_Tree<FLOAT_TYPE>* cct= Tree_Helper::create_a_tree<FLOAT_TYPE>(USE_CPU_LEAF_APPRXIMATE_QP, data,(FLOAT_TYPE)0.005);
	//Convex_Tree<FLOAT_TYPE>* cct= Tree_Helper::create_a_tree<FLOAT_TYPE>(USE_CPU_RECURSIVE_APPRXIMATE_QP, data,(FLOAT_TYPE)0.005);
	Convex_Tree<FLOAT_TYPE>* cct = Tree_Helper::create_a_tree<FLOAT_TYPE>(USE_GPU_LEAF_APPRXIMATE_QP, data, leaf_percent);
	//Convex_Tree<FLOAT_TYPE>* cct= Tree_Helper::create_a_tree<FLOAT_TYPE>(USE_GPU_RECURSIVE_APPRXIMATE_QP, data,(FLOAT_TYPE)0.001);
	cct->print_tree_info();
	cct->save_tree_info(log_file);
	CYW_TIMER timer;
	timer.start_my_timer();
	for (int j = 0; j < batch_num; j++){
		/*-------------------------------- Select Data Points From Dataset Randomly -------------------------------*/
		int test_num = test_num_each_batch;
		//char* idx_filename="exp_random_idx_2w(buffer kd tree).txt";
		//float* xTest_idx=read_data(idx_filename,&dim1,&n_size1);
		//FILE* ran_idx_file= fopen("exp_random_idx_2w(convex_tree).txt","a+");
		//write_file_log("\n",ran_idx_file);
		/*
		Vector<int> indx(test_num);
		for (int i = 0; i < test_num; i++){
			//randomly get an index \in [0,n_size-1]
			int idx = rand();
			indx[i] = idx%data_size;
		}
		*/

		//Matrix<FLOAT_TYPE> q_points = data.extractRows(indx);
		cct->brute_force_kNN(data, K);
		cct->print_kNN_reult(data);
		//q_points[1282].print();
		//if ((j % 5) == 0)
		//	std::cout << "j=" << j << "\n";
		/*------------------------------ Select Data Points From Dataset Randomly ---------------------------------*/
	}
	timer.stop_my_timer();
	char buffer[1000];
	memset(buffer, 0, sizeof(buffer));
	strcat(buffer, "\n*-----------------------------------------KNN QUERY RESULT ----------------------------------------\n");
	strcat(buffer, "*     Query points number:");
	my_strcat(buffer, batch_num*test_num_each_batch);
	strcat(buffer, ", and K=");
	my_strcat(buffer, K);
	strcat(buffer, "\n*     brute force GPU ");
	timer.strcat_to_buffer(buffer);
	strcat(buffer, "\n*-----------------------------------------KNN QUERY RESULT-----------------------------------------");
	write_file_log(buffer, log_file);
	//cct->print_kNN_reult();


	//fclose(log_file);
	//std::cout<<"\n checking....";
	//Vector<NNResult<FLOAT_TYPE>* > results=brute_force_check(data,q_points, K, cct);
	delete cct;
}


Vector<NNResult<FLOAT_TYPE>* > cpu_brute_force(Matrix<FLOAT_TYPE>& data,Matrix<FLOAT_TYPE>& q_points,int K){
	//std::cout<<"\n cpu brute force is starting...." ;
	int dim = data.ncols();
	//long start_t = clock();//GetTickCount();
	Vector<NNResult<FLOAT_TYPE>* > results;
	results.resize(q_points.nrows());

	CYW_TIMER timer;
	timer.start_my_timer();
	for (int i = 0; i < q_points.nrows(); i++){
		FLOAT_TYPE *q = q_points.get_matrix_raw_data() + i*dim;
		Vector<FLOAT_TYPE> tmp_dists_square = pdist2_squre(data, q, dim);
		results[i] = new NNResult<FLOAT_TYPE>();
		results[i]->init(K);
		for (int j = 0; j < tmp_dists_square.size(); j++){
			results[i]->update(tmp_dists_square[j], j);
		}
		std::cout << "\ni=" << i << " ";
		results[i]->print();
	}
	timer.stop_my_timer();
	std::cout << "\n CPU brute force finished,running time is " << timer.get_my_timer() << " s" << endl;
	//long end_t = clock();//GetTickCount();
	//std::cout<<"\n brute force finished, running time is "<< end_t-start_t<<"\ms" ;
	return results;
		
}


void test_cpu_brute_force_kNN_on_real_data_2(int K, char *data_file_name, int test_num_each_batch, int batch_num, FILE* log_file){
	int dim;
	int data_size;
	float *raw_data = read_data(data_file_name, " ", &dim, &data_size);
	write_file_log("\n\n", log_file);
	write_file_log(data_file_name, log_file);
	Matrix<FLOAT_TYPE> data(raw_data, data_size, dim);
	//data /= 1000;
	CYW_TIMER timer;
	timer.start_my_timer();
	/*
	for (int i = 0; i < batch_num; i++){
		Vector<int> indx(test_num_each_batch);
		for (int j = 0; j < test_num_each_batch; j++){
			//randomly get an index \in [0,n_size-1]
			int idx = rand();
			indx[j] = idx%data_size;
		}
		//CYW_TIMER timer1;
		Matrix<FLOAT_TYPE> q_points = data.extractRows(indx);
		cpu_brute_force(data, q_points, K);
		std::cout << "batch = " << i << "\n";
	}
	*/
	cpu_brute_force(data, data, K);
	timer.stop_my_timer();

	char buffer[1000];
	memset(buffer, 0, sizeof(buffer));
	strcat(buffer, "\n*-----------------------------------------KNN QUERY RESULT ----------------------------------------\n");
	strcat(buffer, "*     Query points number:");
	my_strcat(buffer, batch_num*test_num_each_batch);
	strcat(buffer, ", and K=");
	my_strcat(buffer, K);
	strcat(buffer, "\n*     brute force CPU ");
	timer.strcat_to_buffer(buffer);
	strcat(buffer, "\n*-----------------------------------------KNN QUERY RESULT-----------------------------------------");
	write_file_log(buffer, log_file);
}


void test_cpu_brute_force_kNN_on_real_data_1(int K_NN, char *data_file_name, int test_num, FILE* log_file){
	int dim;
	int data_size;
	float *raw_data = read_data(data_file_name, " ", &dim, &data_size);
	float *raw_data_with_index = read_data_add_index(data_file_name, " ", &dim, &data_size);
	write_file_log("\n", log_file);
	write_file_log(data_file_name, log_file);

	CYW_TIMER timer;
	timer.start_my_timer();
	float *KNN_index_with_dist = new float[2*K_NN*test_num];
	float *temp = new float[2*K_NN];

	int p1_index, p2_index, tmp;
	float *p1, *p2;
	float d, max_dist, max_idx;
	for (int i = 0; i < test_num; i++){
		p1_index = *(raw_data_with_index + (i + 1)*(dim + 1) - 1);
		p1 = raw_data_with_index + i*(dim + 1);
		for (int j = 0; j < test_num; j++){
			p2_index = *(raw_data_with_index + (j + 1)*(dim + 1) - 1);
			p2 = raw_data_with_index + j*(dim + 1);
			d = distance(p1, p2, dim);
			if (j < K_NN){
				temp[j * 2] = j;
				temp[j * 2 + 1] = d;
			}
			if (j >= K_NN){
				tmp = 0;
				max_idx = temp[0];
				max_dist = temp[1];
				for (int k = 1; k < K_NN; k++){
					if (temp[2 * k + 1] > max_dist){
						tmp = k;
						max_idx = temp[2 * k];
						max_dist = temp[2 * k + 1];
					}
				}
				if (d < max_dist){
					temp[tmp * 2] = j;
					temp[tmp * 2 + 1] = d;
				}
			}
		}
		memcpy(KNN_index_with_dist + i * 2 * K_NN, temp, (2 * K_NN)*sizeof(float));
	}
	//print the K-nearest neighbor of all points
	for (int i = 0; i < 100; i++){
		std::cout << i << " KNN is: ";
		for (int j = 0; j < K_NN; j++){
			std::cout << KNN_index_with_dist[i*K_NN*2 + j*2] << ", ";
		}
		std::cout << endl;
	}

	timer.stop_my_timer();
	timer.print("brute force on cpu: ");
	char buffer[1000];
	memset(buffer, 0, sizeof(buffer));
	strcat(buffer, "\n*-----------------------------------------KNN QUERY RESULT ----------------------------------------\n");
	strcat(buffer, "*     Query points number:");
	my_strcat(buffer, test_num);
	strcat(buffer, ", and K=");
	my_strcat(buffer, K_NN);
	strcat(buffer, "\n*     brute force CPU ");
	timer.strcat_to_buffer(buffer);
	strcat(buffer, "\n*-----------------------------------------KNN QUERY RESULT-----------------------------------------");
	write_file_log(buffer, log_file);
	free(raw_data);
	free(raw_data_with_index);
}


void test_gpu_KNN(char *data_file_name, char *query_data_file, int K, FLOAT_TYPE leaf_percent, FILE * log_file){
	write_file_log("\n-------------------------------------------------------------------------------------\n", log_file);
	write_file_log(data_file_name, log_file);
	int dim, data_size;
	float *raw_data = read_data(data_file_name, " ", &dim, &data_size);
	Matrix<FLOAT_TYPE> data(raw_data, data_size, dim);

	Convex_Tree<FLOAT_TYPE> *cct = Tree_Helper::create_a_tree<FLOAT_TYPE>(USE_GPU_LEAF_APPRXIMATE_QP, data, leaf_percent);
	cct->print_tree_info();
	cct->save_tree_info(log_file);

	int new_dim, query_data_size;
	float *query_data = read_data(query_data_file, " ", &new_dim, &query_data_size);
	Matrix<FLOAT_TYPE> query_points(query_data, query_data_size, new_dim);

	//cct->brute_force_kNN(query_points, K);
	cct->kNN(query_points, K);
	cct->print_kNN_running_time_info();
	cct->save_KNN_running_time_info(log_file,query_data_size);
	cct->print_kNN_reult(query_points);

}


void test_gpu_KNN_imp(char *data_file_name, char *query_data_file, int K, FLOAT_TYPE leaf_percent, FLOAT_TYPE sample_percent, FILE * log_file){
	write_file_log("\n-------------------------------------------------------------------------------------\n", log_file);
	write_file_log(data_file_name, log_file);
	int dim, data_size;
	float *raw_data = read_data(data_file_name, " ", &dim, &data_size);
	Matrix<FLOAT_TYPE> data(raw_data, data_size, dim);
	cout << "get raw data ending......" << endl;

	int new_dim, query_data_size;
	float *query_data = read_data(query_data_file, " ", &new_dim, &query_data_size);
	Matrix<FLOAT_TYPE> query_points(query_data, query_data_size, new_dim);
	cout << "get query data ending......" << endl;

	int sample_num = floor(data_size * sample_percent);
	Vector<int> sample_index(sample_num);
	int level = 1 / sample_percent;
	for (int i = 0; i < sample_num; i++){
		int index = rand() % level;
		sample_index[i] = index + i * level;
	}
	Matrix<FLOAT_TYPE> sample_points = data.extractRows(sample_index);

	int new_data_num = data_size - sample_num;
	Vector<int> new_data_index(new_data_num);
	int count = 0;
	for (int i = 0; i < sample_num; i++){
		int s_index = sample_index[i] - i * level;
		for (int j = 0; j < level; j++){
			if (j == s_index)  continue;
			else{
				new_data_index[count] = j + i * level;
				count++;
			}
		}
	}
	for (int i = sample_num * level; i < data_size && count < new_data_num; i++){
		new_data_index[count] = i;
		count++;
	}
	Matrix<FLOAT_TYPE> new_data = data.extractRows(new_data_index);
	
	Convex_Tree<FLOAT_TYPE> *sct = Tree_Helper::create_a_tree<FLOAT_TYPE>(USE_GPU_LEAF_APPRXIMATE_QP, sample_points, leaf_percent);
	sct->print_tree_info();
	sct->save_tree_info(log_file);

	sct->kNN(query_points, K);
	sct->update_KNN_index(sample_index, query_data_size);
	sct->save_KNN_running_time_info(log_file, query_data_size);
	
	Convex_Tree<FLOAT_TYPE> *cct = Tree_Helper::create_a_tree<FLOAT_TYPE>(USE_GPU_LEAF_APPRXIMATE_QP, new_data, leaf_percent);
	cct->print_tree_info();
	cct->save_tree_info(log_file);

	int* remain_index = new_data_index.get_matrix_raw_data();
	cct->new_kNN(query_points, K, sct->get_kNN_dists_squre(), sct->get_kNN_indexes(),remain_index);
	cct->new_save_KNN_running_time_info(log_file, query_data_size);
	cct->print_kNN_reult(query_points);

}


int main(int argc, char *const argv[])
{
	char *data_file_name = "data/pam_uni.txt";
	//char *data_file_name = "data/house_uni.txt";
	//char *data_file_name = "data/USCensus1990_uni.txt";

	char *new_data_file_name = "data/new_pam_uni_20W.txt";
	//char *new_data_file_name = "data/new_house_uni_20W.txt";
	//char *new_data_file_name = "data/new_USCensus1990_uni_20w.txt";

	int K = 10;
	FLOAT_TYPE sample_percent = 0.1;
	int query_num = 200000;

	FLOAT_TYPE leaf_percent[8] = {0.01,0.008,0.006,0.004,0.002,0.0009,0.0005,0.0001};
	
	//get_random_data(new_data_file_name,data_file_name, query_num, 0.01);
	//get_raw_data_with_noise(new_data_file_name, data_file_name, query_num, 100);

	cout << "start experiment!" << endl;
	
	/*
	FILE* log_file = fopen("exp_log_brute_force_kNN(cpu).txt","a+");
	test_cpu_brute_force_kNN_on_real_data_1( K, data_file_name, 78095, log_file);
	//test_cpu_brute_force_kNN_on_real_data_2( K, data_file_name, 20, 1, log_file);
	fclose(log_file);
	*/

	/*
	FILE* log_file = fopen("exp_log_brute_force_kNN(gpu).txt", "a+");
	test_gpu_brute_force_kNN_on_real_data(K, data_file_name, 20, 1, 0.15, log_file);
	fclose(log_file);
	*/

	FILE * log_file = fopen("exp_log_kNN(gpu_pam_20W).txt", "a+");
	test_gpu_KNN(data_file_name, new_data_file_name, K, 0.001, log_file);

	//FILE * log_file = fopen("exp_log_kNN(gpu_improved_pam_20W_10%).txt", "a+");
	//test_gpu_KNN_imp(data_file_name, new_data_file_name, K, 0.001, sample_percent, log_file);
	
	fclose(log_file);

	return 0;
}

