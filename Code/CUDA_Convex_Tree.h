/*
*
* Copyright (C) 2017-  Yewang Chen<ywchen@hqu.edu.cn;nalandoo@gmail.com>
* License: GPL v1
* This software may be modified and distributed under the terms
* of license.
*
*/
#ifndef CUDA_CONVEX_TREE_H 
    #define CUDA_CONVEX_TREE_H
#endif
#include "Convex_Tree.h"
#include "ConvexNode.h"
#include "basic_functions.h"
#include "cyw_timer.h"
#include "cyw_types.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"


template <class T>
class CUDA_Convex_Tree :public Convex_Tree<T>
{
public:
	CUDA_Convex_Tree();
	virtual ~CUDA_Convex_Tree(){
		delete[] this->dist_square_k_mins_global;
		delete[] this->idx_k_mins_global;
		delete[] this->dist_computation_times_arr;
		delete[] this->quadprog_times_arr;
	};
	//leaf_pts_percent : the percent that specify the number of leaf_pts_num
	CUDA_Convex_Tree(Matrix<T>& data, FLOAT_TYPE leaf_pts_percent, int alg_type);

	//after kNN processing, get the result by the following three entries
	//they are all virtual procedures, and override here.
	virtual FLOAT_TYPE*  get_kNN_dists(int query_point_index){
		//return this->
		return 0;
	};

	virtual FLOAT_TYPE*  get_kNN_dists_squre(int query_point_index){
		return this->dist_square_k_mins_global + query_point_index*(this->K);
	};

	virtual  int* get_kNN_indexes(int query_point_index){
		return this->idx_k_mins_global + query_point_index* (this->K);
	};

	virtual FLOAT_TYPE*  get_kNN_dists_squre(){
		return this->dist_square_k_mins_global;
	};

	virtual int* get_kNN_indexes(){
		return this->idx_k_mins_global;
	};

	virtual void print_kNN_running_time_info();
	virtual void save_KNN_running_time_info(FILE* log_file,int query_num);
	virtual void new_save_KNN_running_time_info(FILE* log_file, int query_num);
	virtual void update_KNN_index(Vector<int> &data_index, int query_num);
	//virtual void set_batch_size(int batch_size);

protected:
	//dist_k_mins_global and idx_k_mins_global will be initialize in 'kNN_XXX()'
	FLOAT_TYPE* dist_square_k_mins_global = NULL;
	int  *idx_k_mins_global = NULL;
	long *dist_computation_times_arr = NULL;
	long *quadprog_times_arr = NULL;
	long *dist_computation_times_in_quadprog_arr = NULL;

	//sum of all distance computation times saved in computation_statistic_num[0,2,4....]
	long long total_dist_computation_times = 0;
	//sum of all quadprog times saved in computation_statistic_num[1,3,5....]
	long total_quadprog_times = 0;
	long total_dist_computation_times_in_quadprog = 0;

	//override do_kNN defined in super class, and implement specific algorithm here.
	virtual void do_brute_force_and_update_from_outliers(Matrix<T> &query_points_mat);
	//override do_kNN defined in super class, and implement specific algorithm here.
	virtual void do_kNN(Matrix<T> &query_points_mat);
	//override do_kNN defined in super class, and implement specific algorithm here.
	virtual void new_do_kNN(Matrix<T> &query_points_mat,int* remain_index);
	//override do_kNN defined in super class, and implement specific algorithm here.
	virtual void do_brute_force_kNN(Matrix<T> &query_points_mat);
	

	//for each query point, we find its approximate nodes, and save it in appr_leaf_node_indexes.
	void do_find_approximate_nodes(Matrix<T>& query_points_mat);

	/*
	This is virtual procedure, a entry for specific task in subclass.
	it will be call in init_process_query_points defined in  superclass 'Convex_tree'
	*/
	virtual void  init_kNN_result(Matrix<T> &query_points_mat);
	virtual void  new_init_kNN_result(Matrix<T> &query_points_mat, FLOAT_TYPE* dist_square_k_mins_global, int* idx_k_mins_global);
	virtual void  init_kNN(FLOAT_TYPE* dist_square_k_mins_global, int* idx_k_mins_global);

	//override
	virtual void  do_print_kNN_result(Matrix<T>& query_points_mat){

		int num_query_pts =query_points_mat.nrows();
		for (int i = 0; i < 100; i++){
			/*
			std::cout << "\n the dist_squre of kNN of " << i << "{th}:";
			for (int j = 0; j<this->K; j++){
				std::cout << this->dist_square_k_mins_global[i*this->K + j] << ", ";
			}*/
			std::cout << "\n   the indexes of kNN of " << i << "{th} query point:";
			for (int j = 0; j<this->K; j++){
				std::cout << this->idx_k_mins_global[i*this->K + j] << ", ";
			}
		}
	}
private:
	int WORKGROUP_SIZE_BRUTE = 256;
	int BATCH_SIZE_OF_QUERY_POINTS_PUSCH_TO_DEVICE = 100000;

	bool data_set_buffer_allocated = false;
	void* data_set_buffer_for_brute_force;

	/* it is not good to declare the following 2 vars here, please refactoring it next time */
	//it is used in find_a_leaf_node_for_query_point to figure out whether the leaf node is found.
	bool is_leaf_node_found;
	//it is also used in find_a_leaf_node_for_query_point to figure out the leaf node index which is found.
	int cur_leaf_node_found = -1;

	//saves the batches of query points
	int cur_batch_iter = 0;

	//if current_visiting_node_indexes_of_query_points[i]=j, then it means the i^th query point currently visiting the j^th node
	//during find kNN by traversing the tree.
	std::vector<int> current_visiting_node_indexes_of_query_points;

	//init CUDA devices
	//bool init_CUDA_device();

	void init_global_final_NN_result(int query_points_num, int K);

};

template <class T>
CUDA_Convex_Tree<T>::CUDA_Convex_Tree(Matrix<T>& data,
	FLOAT_TYPE leaf_pts_percent,
	int alg_type) :Convex_Tree<T>(data, leaf_pts_percent, alg_type)
{
	//init platforms and devices
	//init_CUDA_device();
	//init parameters 3-14 for train data set.
	//init_openCL_sorted_data_set();

	//init parameters 0-2, and 17 for train data set, here just allocate space for them without writing data
	//init_dynamic_batch_query_points();
}

template <typename T>
void CUDA_Convex_Tree<T>::init_global_final_NN_result(int query_points_num, int K){
	if (this->dist_square_k_mins_global != NULL){
		delete[]this->dist_square_k_mins_global;
		delete[]this->idx_k_mins_global;
		this->dist_square_k_mins_global = NULL;
		this->idx_k_mins_global = NULL;
	}
	this->dist_square_k_mins_global = new FLOAT_TYPE[query_points_num*K];

	this->idx_k_mins_global = new int[query_points_num*K];
	memset(this->idx_k_mins_global, -1, sizeof(int)*query_points_num*K);
}

//for each query point, we find its approximate nodes, and save it in appr_leaf_node_indexes.
template <typename T>
void CUDA_Convex_Tree<T>::do_find_approximate_nodes(Matrix<T>& query_points_mat){
	int query_points_num = query_points_mat.nrows();
	this->timer_total_approximate_searching.start_my_timer();

	this->appr_leaf_node_indexes.resize(query_points_num);
	//for each query point, we find its approximate nodes, and save it in appr_leaf_node_indexes.
	for (int i = 0; i<query_points_num; i++){
		int cur_node_index = 0;
		Vector<T>& q = query_points_mat.extractRow(i);
		while (this->nodes[cur_node_index]->isLeaf == false){
			//this->nodes[cur_node_index]->print()  ;
			Vector<T>& left_center_vec = this->nodes[cur_node_index]->left_center;
			Vector<T>& right_center_vec = this->nodes[cur_node_index]->right_center;
			FLOAT_TYPE left_dist_squre = pdist2_squre(q, left_center_vec);
			FLOAT_TYPE right_dist_squre = pdist2_squre(q, right_center_vec);
			//count the distance computation times.
			this->dist_computation_times_in_host += 2;
			if (left_dist_squre >= right_dist_squre){
				cur_node_index = this->nodes[cur_node_index]->right;
			}
			else{
				cur_node_index = this->nodes[cur_node_index]->left;
			}
		};
		int cur_leaf_index = this->nodes[cur_node_index]->leaf_index;
		this->appr_leaf_node_indexes[i] = cur_leaf_index;
	}
	this->timer_total_approximate_searching.stop_my_timer();
}


//override do_kNN defined in super class, and implement specific algorithm here.
template <typename T>
void CUDA_Convex_Tree<T>::do_brute_force_and_update_from_outliers(Matrix<T> &query_points_mat){
	//---param 1
	int candidate_query_points_num = query_points_mat.nrows();
	
	//---param 2
	int * candidate_query_points_indexes = new int[candidate_query_points_num];
	for (int i = 0; i < candidate_query_points_num; i++){
		candidate_query_points_indexes[i] = i;
	}

	//---param 3
	FLOAT_TYPE* candidate_query_points_set = query_points_mat.get_matrix_raw_data();

	//---param 4
	int all_exc_num = this->all_exc_data.nrows();

	//---param 5
	FLOAT_TYPE* outliers_set = this->all_exc_data.get_matrix_raw_data();
	//this->all_exc_indexes.print("\nthe index of outliers:");

	//---param 5
	int* outliers_index = this->all_exc_indexes.get_matrix_raw_data();

	//---param 7
	int K_NN = this->K;

	//---param 8
	int DIM = this->all_exc_data.ncols();

	std::cout << "\ncall kernel brute force for outliers...." << endl;;

	call_cuda_kernel_brute_force_for_outliers( candidate_query_points_num,
											   candidate_query_points_indexes,
											   candidate_query_points_set,
											   all_exc_num,
											   outliers_set,
											   outliers_index,
											   this->dist_square_k_mins_global,
											   this->idx_k_mins_global,
											   K_NN,
											   DIM);

}

/*
//override do_kNN defined in super class, and implement specific algorithm here.
template <typename T>
void CUDA_Convex_Tree<T>::do_brute_force_and_update_from_outliers(Matrix<T> &query_points_mat){
	int K_NN = this->K;
	int all_exc_num = this->all_exc_data.nrows();
	int dim = this->all_exc_data.ncols();
	FLOAT_TYPE* outliers_set = this->all_exc_data.get_matrix_raw_data();
	FLOAT_TYPE* candidate_query_points_set = query_points_mat.get_matrix_raw_data();
	int candidate_query_points_num = query_points_mat.nrows();
	FLOAT_TYPE dist_square_tmp = 0, tmp = 0;
	int tmp_idx = 0;

	for (int i = 0; i < candidate_query_points_num; i++){
		for (int j = 0; j < all_exc_num; j++){
			dist_square_tmp = pdist2_squre(candidate_query_points_set + i * dim, outliers_set + j * dim, dim);
			
			//get the current k^th min_dist_square of current query point
			FLOAT_TYPE cur_k_min_dist_square = this->dist_square_k_mins_global[(i+1)*K_NN - 1];

			if (cur_k_min_dist_square > dist_square_tmp){
				int k = K_NN - 1;
				this->dist_square_k_mins_global[i*K_NN + k] = dist_square_tmp;
				int pts_idx = this->all_exc_indexes.extract(j);
				this->idx_k_mins_global[i*K_NN + k] = pts_idx;
				for (; k > 0; k--){
					if (this->dist_square_k_mins_global[i*K_NN + k - 1] > this->dist_square_k_mins_global[i*K_NN + k]){
						//swap dist
						tmp = this->dist_square_k_mins_global[i*K_NN + k - 1];
						this->dist_square_k_mins_global[i*K_NN + k - 1] = this->dist_square_k_mins_global[i*K_NN + k];
						this->dist_square_k_mins_global[i*K_NN + k] = tmp;
						//swap indexes
						tmp_idx = this->idx_k_mins_global[i*K_NN + k - 1];
						this->idx_k_mins_global[i*K_NN + k - 1] = this->idx_k_mins_global[i*K_NN + k];
						this->idx_k_mins_global[i*K_NN + k] = tmp_idx;
					}
					else break;
				}
			}
		}
	}
}
*/

//override do_kNN defined in super class, and implement specific algorithm here.
template <typename T>
void CUDA_Convex_Tree<T>::do_kNN(Matrix<T> &query_points_mat){
	int dim = query_points_mat.ncols();
	std::cout << "\npreparing parameters....\n";
	//---param 1
	int query_points_num = query_points_mat.nrows();
	std::cout << "query_points_num: " << query_points_num << endl;
	
	//---param 2
	int * candidate_query_points_indexes = new int[query_points_num];
	for (int i = 0; i < query_points_num; i++){
		candidate_query_points_indexes[i] = i;
	}

	//---param 3
	FLOAT_TYPE *candidate_query_points_set = query_points_mat.get_matrix_raw_data();
	
	//---param 4
	int *candidate_query_points_appr_leaf_node_indexes = this->appr_leaf_node_indexes.get_matrix_raw_data();
	/*
	std::cout << "candiate query points approximate leaf node indexes: ";
	for (int i = 0; i < 10; i++){
		std::cout << *(candidate_query_points_appr_leaf_node_indexes + i) << " ";
	}*/

	//---param 5
	int sorted_data_len = this->m_sorted_data.nrows();

	//---param 6
	FLOAT_TYPE* all_sorted_data_set = this->m_sorted_data.get_matrix_raw_data();
	//std::cout << "\nthe length of m_sorted_data: " << this->m_sorted_data.nrows();

	//---param 7
	int* sorted_data_set_indexes = this->m_sorted_data_ori_indexes.get_matrix_raw_data();
	
	//print sorted data set indexes
	//sort(sorted_data_set_indexes, sorted_data_set_indexes + this->m_sorted_data.nrows());
	/*std::cout << "\nsorted data set indexes:";
	for (int i = 0,j = 0; i < this->m_sorted_data.nrows(); i++, j++){
		if (j%10 == 0)  std::cout << endl;
		std::cout << sorted_data_set_indexes[i] << " ";
	}*/
	

	//---param 8
	int tree_nodes_num = this->nodes_num;

	//---param 9
	CONVEX_TREE *tree_struct = this->simplified_tree.get_matrix_raw_data();


	//---param 10
	int all_leaf_nodes_constraint_num = 0;

	int cur_leaf_node_ori_index;
	for (int i = 0; i<this->leaf_nodes_num; i++){
		cur_leaf_node_ori_index = this->leaf_nodes_ori_indexes[i];
		all_leaf_nodes_constraint_num += this->nodes[cur_leaf_node_ori_index]->ALPHA.nrows();
	}
	//---param 11: all_ALPHA_set_buffer
	FLOAT_TYPE* all_leaf_nodes_ALPHA_set = new FLOAT_TYPE[all_leaf_nodes_constraint_num*dim];
	int cur_pos = 0;
	//---the first node is root who has no alpha and beta
	for (int i = 0; i<this->leaf_nodes_num; i++){
		cur_leaf_node_ori_index = this->leaf_nodes_ori_indexes[i];
		int tmp_num = this->nodes[cur_leaf_node_ori_index]->ALPHA.nrows();
		//---copy data
		memcpy(all_leaf_nodes_ALPHA_set + cur_pos*dim, this->nodes[cur_leaf_node_ori_index]->ALPHA.get_matrix_raw_data(), tmp_num*dim*sizeof(FLOAT_TYPE));
		cur_pos += tmp_num;
	}

	//---param 12: all_BETA_set_buffer
	FLOAT_TYPE* all_leaf_nodes_BETA_set = new FLOAT_TYPE[all_leaf_nodes_constraint_num];
	cur_pos = 0;
	//---the first node is root who has no alpha and beta
	for (int i = 0; i<this->leaf_nodes_num; i++){
		cur_leaf_node_ori_index = this->leaf_nodes_ori_indexes[i];
		int tmp_num = this->nodes[cur_leaf_node_ori_index]->BETA.size();
		//---copy data
		memcpy(all_leaf_nodes_BETA_set + cur_pos, this->nodes[cur_leaf_node_ori_index]->BETA.get_matrix_raw_data(), tmp_num*sizeof(FLOAT_TYPE));
		cur_pos += tmp_num;
	}

	//---param 13: all_constrains_num_of_each_leaf_nodes_buffer
	int* all_constrains_num_of_each_leaf_nodes = new int[this->leaf_nodes_num];
	cur_pos = 0;
	for (int i = 0; i<this->leaf_nodes_num; i++){
		cur_leaf_node_ori_index = this->leaf_nodes_ori_indexes[i];
		int tmp_num = this->nodes[cur_leaf_node_ori_index]->BETA.size();
		all_constrains_num_of_each_leaf_nodes[i] = tmp_num;
	}


	//---param 14: all_nodes_offsets_in_all_ALPHA_buffer
	int* all_leaf_nodes_offsets_in_all_ALPHA = new int[this->leaf_nodes_num];
	cur_pos = 0;
	for (int i = 0; i<this->leaf_nodes_num; i++){
		cur_leaf_node_ori_index = this->leaf_nodes_ori_indexes[i];
		int tmp_num = this->nodes[cur_leaf_node_ori_index]->ALPHA.nrows();
		all_leaf_nodes_offsets_in_all_ALPHA[i] = cur_pos;
		cur_pos += tmp_num;
	}

	//---param 15: leaf_node_num (this int type, unnecessary to create buffer)

	//---param 16: ancestor_nodes_ids_buffer
	//---all_alpha_num
	int* all_leaf_nodes_ancestor_nodes_ids = new int[all_leaf_nodes_constraint_num];

	cur_pos = 0;
	//---the first node is root who has no alpha and beta
	for (int i = 0; i<this->leaf_nodes_num; i++){
		cur_leaf_node_ori_index = this->leaf_nodes_ori_indexes[i];
		int tmp_num = this->nodes[cur_leaf_node_ori_index]->ancestor_nodes_num;
		//---copy data: here, all_leaf_nodes_ancestor_nodes_ids has a shift, because the root node has no constraint, and
		//------------  each node has different alpha from its brother node, e.g., suppose node_i and node_j are
		//------------  brother nodes with K constraints, the fist (K-1) alpha of node_i and  and node_j are the
		//------------  same, but the last alpha of the two nodes are different, in fact:
		//------------  -alpha_k of node_i= alpha_k of node_j .
		for (int j = 0; j<tmp_num - 1; j++){
			all_leaf_nodes_ancestor_nodes_ids[cur_pos + j] = this->nodes[cur_leaf_node_ori_index]->ancestor_node_ids[j + 1];
		}
		all_leaf_nodes_ancestor_nodes_ids[cur_pos + tmp_num - 1] = cur_leaf_node_ori_index;
		cur_pos += tmp_num;
	}

	//param 17:
	int* sorted_leaf_nodes_start_pos_in_sorted_data = new int[this->leaf_nodes_num];
	for (int i = 0; i<this->leaf_nodes_num; i++){
		sorted_leaf_nodes_start_pos_in_sorted_data[i] = this->leaf_nodes_start_pos_in_data_set[i];
	}
	

	//param 18
	int* pts_num_in_sorted_leaf_nodes = new int[this->leaf_nodes_num];
	for (int i = 0; i<this->leaf_nodes_num; i++){
		cur_leaf_node_ori_index = this->leaf_nodes_ori_indexes[i];
		pts_num_in_sorted_leaf_nodes[i] = this->nodes[cur_leaf_node_ori_index]->pts_number;
	}

	int NODES_NUM = this->nodes_num;
	int DIM = dim;

	//this->do_print_kNN_result(query_points_mat);

	std::cout << "\ncall kernel....";

	call_cuda_kernel(
		query_points_num,
		candidate_query_points_indexes,
		candidate_query_points_set,
		candidate_query_points_appr_leaf_node_indexes,
		sorted_data_len,
		all_sorted_data_set,
		sorted_data_set_indexes,
		tree_nodes_num,
		tree_struct,
		all_leaf_nodes_constraint_num,
		all_leaf_nodes_ALPHA_set,
		all_leaf_nodes_BETA_set,
		all_constrains_num_of_each_leaf_nodes,
		all_leaf_nodes_offsets_in_all_ALPHA,
		this->leaf_nodes_num,
		all_leaf_nodes_ancestor_nodes_ids,
		sorted_leaf_nodes_start_pos_in_sorted_data,
		pts_num_in_sorted_leaf_nodes,
		this->dist_square_k_mins_global,
		this->idx_k_mins_global,
		this->K,
		this->dist_computation_times_arr,
		this->quadprog_times_arr,
		this->dist_computation_times_in_quadprog_arr,
		NODES_NUM,
		DIM
		);
	printf("\nCUDA_Convex_Tree::do_KNN over...");
}

//override do_kNN defined in super class, and implement specific algorithm here.
template <typename T>
void CUDA_Convex_Tree<T>::new_do_kNN(Matrix<T> &query_points_mat,int* remain_index){
	int dim = query_points_mat.ncols();
	std::cout << "\npreparing parameters....\n";
	//---param 1
	int query_points_num = query_points_mat.nrows();
	std::cout << "query_points_num: " << query_points_num << endl;

	//---param 2
	int * candidate_query_points_indexes = new int[query_points_num];
	for (int i = 0; i < query_points_num; i++){
		candidate_query_points_indexes[i] = i;
	}

	//---param 3
	FLOAT_TYPE *candidate_query_points_set = query_points_mat.get_matrix_raw_data();

	//---param 4
	//int *candidate_query_points_appr_leaf_node_indexes = this->appr_leaf_node_indexes.get_matrix_raw_data();
	/*
	std::cout << "candiate query points approximate leaf node indexes: ";
	for (int i = 0; i < 10; i++){
	std::cout << *(candidate_query_points_appr_leaf_node_indexes + i) << " ";
	}*/

	//---param 5
	int sorted_data_len = this->m_sorted_data.nrows();

	//---param 6
	FLOAT_TYPE* all_sorted_data_set = this->m_sorted_data.get_matrix_raw_data();
	//std::cout << "\nthe length of m_sorted_data: " << this->m_sorted_data.nrows();

	//---param 7
	int* sorted_data_set_indexes = this->m_sorted_data_ori_indexes.get_matrix_raw_data();

	//print sorted data set indexes
	//sort(sorted_data_set_indexes, sorted_data_set_indexes + this->m_sorted_data.nrows());
	/*std::cout << "\nsorted data set indexes:";
	for (int i = 0,j = 0; i < this->m_sorted_data.nrows(); i++, j++){
	if (j%10 == 0)  std::cout << endl;
	std::cout << sorted_data_set_indexes[i] << " ";
	}*/


	//---param 8
	int tree_nodes_num = this->nodes_num;

	//---param 9
	CONVEX_TREE *tree_struct = this->simplified_tree.get_matrix_raw_data();


	//---param 10
	int all_leaf_nodes_constraint_num = 0;

	int cur_leaf_node_ori_index;
	for (int i = 0; i<this->leaf_nodes_num; i++){
		cur_leaf_node_ori_index = this->leaf_nodes_ori_indexes[i];
		all_leaf_nodes_constraint_num += this->nodes[cur_leaf_node_ori_index]->ALPHA.nrows();
	}
	//---param 11: all_ALPHA_set_buffer
	FLOAT_TYPE* all_leaf_nodes_ALPHA_set = new FLOAT_TYPE[all_leaf_nodes_constraint_num*dim];
	int cur_pos = 0;
	//---the first node is root who has no alpha and beta
	for (int i = 0; i<this->leaf_nodes_num; i++){
		cur_leaf_node_ori_index = this->leaf_nodes_ori_indexes[i];
		int tmp_num = this->nodes[cur_leaf_node_ori_index]->ALPHA.nrows();
		//---copy data
		memcpy(all_leaf_nodes_ALPHA_set + cur_pos*dim, this->nodes[cur_leaf_node_ori_index]->ALPHA.get_matrix_raw_data(), tmp_num*dim*sizeof(FLOAT_TYPE));
		cur_pos += tmp_num;
	}

	//---param 12: all_BETA_set_buffer
	FLOAT_TYPE* all_leaf_nodes_BETA_set = new FLOAT_TYPE[all_leaf_nodes_constraint_num];
	cur_pos = 0;
	//---the first node is root who has no alpha and beta
	for (int i = 0; i<this->leaf_nodes_num; i++){
		cur_leaf_node_ori_index = this->leaf_nodes_ori_indexes[i];
		int tmp_num = this->nodes[cur_leaf_node_ori_index]->BETA.size();
		//---copy data
		memcpy(all_leaf_nodes_BETA_set + cur_pos, this->nodes[cur_leaf_node_ori_index]->BETA.get_matrix_raw_data(), tmp_num*sizeof(FLOAT_TYPE));
		cur_pos += tmp_num;
	}

	//---param 13: all_constrains_num_of_each_leaf_nodes_buffer
	int* all_constrains_num_of_each_leaf_nodes = new int[this->leaf_nodes_num];
	cur_pos = 0;
	for (int i = 0; i<this->leaf_nodes_num; i++){
		cur_leaf_node_ori_index = this->leaf_nodes_ori_indexes[i];
		int tmp_num = this->nodes[cur_leaf_node_ori_index]->BETA.size();
		all_constrains_num_of_each_leaf_nodes[i] = tmp_num;
	}


	//---param 14: all_nodes_offsets_in_all_ALPHA_buffer
	int* all_leaf_nodes_offsets_in_all_ALPHA = new int[this->leaf_nodes_num];
	cur_pos = 0;
	for (int i = 0; i<this->leaf_nodes_num; i++){
		cur_leaf_node_ori_index = this->leaf_nodes_ori_indexes[i];
		int tmp_num = this->nodes[cur_leaf_node_ori_index]->ALPHA.nrows();
		all_leaf_nodes_offsets_in_all_ALPHA[i] = cur_pos;
		cur_pos += tmp_num;
	}

	//---param 15: leaf_node_num (this int type, unnecessary to create buffer)

	//---param 16: ancestor_nodes_ids_buffer
	//---all_alpha_num
	int* all_leaf_nodes_ancestor_nodes_ids = new int[all_leaf_nodes_constraint_num];

	cur_pos = 0;
	//---the first node is root who has no alpha and beta
	for (int i = 0; i<this->leaf_nodes_num; i++){
		cur_leaf_node_ori_index = this->leaf_nodes_ori_indexes[i];
		int tmp_num = this->nodes[cur_leaf_node_ori_index]->ancestor_nodes_num;
		//---copy data: here, all_leaf_nodes_ancestor_nodes_ids has a shift, because the root node has no constraint, and
		//------------  each node has different alpha from its brother node, e.g., suppose node_i and node_j are
		//------------  brother nodes with K constraints, the fist (K-1) alpha of node_i and  and node_j are the
		//------------  same, but the last alpha of the two nodes are different, in fact:
		//------------  -alpha_k of node_i= alpha_k of node_j .
		for (int j = 0; j<tmp_num - 1; j++){
			all_leaf_nodes_ancestor_nodes_ids[cur_pos + j] = this->nodes[cur_leaf_node_ori_index]->ancestor_node_ids[j + 1];
		}
		all_leaf_nodes_ancestor_nodes_ids[cur_pos + tmp_num - 1] = cur_leaf_node_ori_index;
		cur_pos += tmp_num;
	}

	//param 17:
	int* sorted_leaf_nodes_start_pos_in_sorted_data = new int[this->leaf_nodes_num];
	for (int i = 0; i<this->leaf_nodes_num; i++){
		sorted_leaf_nodes_start_pos_in_sorted_data[i] = this->leaf_nodes_start_pos_in_data_set[i];
	}


	//param 18
	int* pts_num_in_sorted_leaf_nodes = new int[this->leaf_nodes_num];
	for (int i = 0; i<this->leaf_nodes_num; i++){
		cur_leaf_node_ori_index = this->leaf_nodes_ori_indexes[i];
		pts_num_in_sorted_leaf_nodes[i] = this->nodes[cur_leaf_node_ori_index]->pts_number;
	}

	int NODES_NUM = this->nodes_num;
	int DIM = dim;

	//this->do_print_kNN_result(query_points_mat);

	std::cout << "\ncall kernel....";

	new_call_cuda_kernel(
		query_points_num,
		candidate_query_points_indexes,
		candidate_query_points_set,
		sorted_data_len,
		all_sorted_data_set,
		sorted_data_set_indexes,
		tree_nodes_num,
		tree_struct,
		all_leaf_nodes_constraint_num,
		all_leaf_nodes_ALPHA_set,
		all_leaf_nodes_BETA_set,
		all_constrains_num_of_each_leaf_nodes,
		all_leaf_nodes_offsets_in_all_ALPHA,
		this->leaf_nodes_num,
		all_leaf_nodes_ancestor_nodes_ids,
		sorted_leaf_nodes_start_pos_in_sorted_data,
		pts_num_in_sorted_leaf_nodes,
		this->dist_square_k_mins_global,
		this->idx_k_mins_global,
		remain_index,
		this->K,
		this->dist_computation_times_arr,
		this->quadprog_times_arr,
		this->dist_computation_times_in_quadprog_arr,
		NODES_NUM,
		DIM
		);
	printf("\nCUDA_Convex_Tree::do_KNN over...");
}


//override do_brute_force_kNN defined in super class, and implement specific algorithm here.
template <typename T>
void CUDA_Convex_Tree<T>::do_brute_force_kNN(Matrix<T> &query_points_mat){
	std::cout << "\nStarting brute force KNN.......\n" << endl;
	int  candidate_query_points_num = query_points_mat.nrows();
	int  *candidate_query_points_indexes = new int[candidate_query_points_num];
	for (int i = 0; i < candidate_query_points_num; i++){
		candidate_query_points_indexes[i] = i;
	};
	FLOAT_TYPE   *candidate_query_points_set = query_points_mat.get_matrix_raw_data();
	int          data_set_size = this->getData().nrows();
	std::cout << "data_set_size:" << data_set_size << endl;
	FLOAT_TYPE   *data_set = this->getData().get_matrix_raw_data();
	int DIM = query_points_mat.ncols();
	//FLOAT_TYPE   *KNN_index_with_dist = new FLOAT_TYPE[2 * (this->K) * data_set_size];

	call_cuda_kernel_brute_force( candidate_query_points_num,
		                          candidate_query_points_indexes,
								  candidate_query_points_set,
								  data_set_size,
								  data_set,
								  this->dist_square_k_mins_global,
								  this->idx_k_mins_global,
								  this->K,
								  DIM);
}

//virtual procedure
template <typename T>
void  CUDA_Convex_Tree<T>::update_KNN_index(Vector<int> &data_index,int query_num){
	int* index = data_index.get_matrix_raw_data();
	for (int i = 0; i < query_num; i++){
		for (int j = 0; j < this->K; j++){
			int raw_index = this->idx_k_mins_global[i*this->K + j];
			//if (raw_index >= this->m_data.nrows())
			//	cout << "out of index!" << endl;
			this->idx_k_mins_global[i*this->K + j] = index[raw_index];
		}
	}
}


//virtual procedure
template <typename T>
void CUDA_Convex_Tree<T>::print_kNN_running_time_info()
{

	std::cout << "\n*-----------------------------------------KNN QUERY RESULT ----------------------------------------\n";
	//std::cout << "*     Query points number:" << this->query_points_vec.size() << ", and K=" << this->K << "\n";
	std::cout << "*     GPU Alorithm ";
	this->timer_whole_alg.print();
	std::cout << "*          Where:\n";
	this->timer_init_process_query_points.print("*          (1) Init query points: ");
	this->timer_total_approximate_searching.print("*              where Approximate searching: ");
	std::cout << "*          (2) Quadratic programming Calls in devices=" << this->total_quadprog_times << " \n";
	std::cout << "*              where inner productions (alpha'*q) in quadprog =" << this->total_dist_computation_times_in_quadprog << " \n";
	std::cout << "*          (3) Distance computations in devices=" << this->total_dist_computation_times << " \n";

	std::cout << "*          (4) Distance computations in approximate searching in host=" << this->dist_computation_times_in_host << " \n";
	std::cout << "*          (5) Overall distance computations in host and devices=(2)+ (3)+(4)=" << this->total_dist_computation_times_in_quadprog + this->total_dist_computation_times + this->dist_computation_times_in_host << " \n";
	std::cout << "*          (6) BATCH_SIZE=" << this->BATCH_SIZE_OF_QUERY_POINTS_PUSCH_TO_DEVICE << ", batch iteration times=" << this->cur_batch_iter << "\n";
	std::cout << "*-----------------------------------------KNN QUERY RESULT ----------------------------------------\n";

};

//virtual procedure
template <typename T>
void CUDA_Convex_Tree<T>::save_KNN_running_time_info(FILE* log_file,int query_num)
{
	char buffer[1000];
	memset(buffer, 0, sizeof(buffer));
	strcat(buffer, "\n*-----------------------------------------KNN QUERY RESULT ----------------------------------------\n");
	strcat(buffer, "*   Query points number:");
	my_strcat(buffer, query_num);
	strcat(buffer, ", and K=");
	my_strcat(buffer, this->K);
	strcat(buffer, "\n*     GPU Alorithm ");
	this->timer_whole_alg.strcat_to_buffer(buffer);

	strcat(buffer, "\n*          Where:");
	this->timer_init_process_query_points.strcat_to_buffer("\n*          (1) Init query points: ", buffer);
	this->timer_total_approximate_searching.strcat_to_buffer("\n*              where Approximate searching: ", buffer);
	this->timer_do_KNN.strcat_to_buffer("\n*          (2) Do KNN: ", buffer);
	strcat(buffer, "\n*          (3) Quadratic programming Calls in devices=");
	my_strcat(buffer, this->total_quadprog_times);
	strcat(buffer, "\n*              where inner productions (alpha'*q) in quadprog =");
	my_strcat(buffer, this->total_dist_computation_times_in_quadprog);
	strcat(buffer, "\n*          (4) Distance computations in devices=");
	my_strcat(buffer, this->total_dist_computation_times);

	strcat(buffer, "\n*          (5) Distance computations in approximate searching in host=");
	my_strcat(buffer, this->dist_computation_times_in_host);
	strcat(buffer, "\n*          (6) Overall distance computations in host and devices=(2)+ (3)+(4)=");
	my_strcat_longlong(buffer, this->total_dist_computation_times_in_quadprog + this->total_dist_computation_times + this->dist_computation_times_in_host);
	strcat(buffer, "\n*          (7) BATCH_SIZE=");
	my_strcat(buffer, this->BATCH_SIZE_OF_QUERY_POINTS_PUSCH_TO_DEVICE);
	strcat(buffer, ", batch iteration times=");
	my_strcat(buffer, this->cur_batch_iter);
	strcat(buffer, "\n*-----------------------------------------KNN QUERY RESULT----------------------------------------");
	write_file_log(buffer, log_file);
};

//virtual procedure
template <typename T>
void CUDA_Convex_Tree<T>::new_save_KNN_running_time_info(FILE* log_file, int query_num)
{
	char buffer[1000];
	memset(buffer, 0, sizeof(buffer));
	strcat(buffer, "\n*-----------------------------------------KNN QUERY RESULT ----------------------------------------\n");
	strcat(buffer, "*   Query points number:");
	my_strcat(buffer, query_num);
	strcat(buffer, ", and K=");
	my_strcat(buffer, this->K);
	strcat(buffer, "\n*     GPU Alorithm ");
	this->timer_whole_alg.strcat_to_buffer(buffer);

	strcat(buffer, "\n*          Where:");
	this->timer_init_process_query_points.strcat_to_buffer("\n*          (1) Init query points: ", buffer);;
	this->timer_do_KNN.strcat_to_buffer("\n*          (2) Do KNN: ", buffer);
	strcat(buffer, "\n*          (3) Quadratic programming Calls in devices=");
	my_strcat(buffer, this->total_quadprog_times);
	strcat(buffer, "\n*              where inner productions (alpha'*q) in quadprog =");
	my_strcat(buffer, this->total_dist_computation_times_in_quadprog);
	strcat(buffer, "\n*          (4) Distance computations in devices=");
	my_strcat(buffer, this->total_dist_computation_times);

	strcat(buffer, "\n*          (5) Distance computations in approximate searching in host=");
	my_strcat(buffer, this->dist_computation_times_in_host);
	strcat(buffer, "\n*          (6) Overall distance computations in host and devices=(2)+ (3)+(4)=");
	my_strcat_longlong(buffer, this->total_dist_computation_times_in_quadprog + this->total_dist_computation_times + this->dist_computation_times_in_host);
	strcat(buffer, "\n*          (7) BATCH_SIZE=");
	my_strcat(buffer, this->BATCH_SIZE_OF_QUERY_POINTS_PUSCH_TO_DEVICE);
	strcat(buffer, ", batch iteration times=");
	my_strcat(buffer, this->cur_batch_iter);
	strcat(buffer, "\n*-----------------------------------------KNN QUERY RESULT----------------------------------------");
	write_file_log(buffer, log_file);
};


/*
This is virtual procedure, a entry for specific task in subclass.
it will be call in init_process_query_points defined in  superclass 'Convex_tree'
*/
template <typename T>
void CUDA_Convex_Tree<T>::init_kNN_result(Matrix<T>& query_points_mat)
{
	//init final result
	int query_points_num = query_points_mat.nrows();
	//init dist_square_k_mins_global and idx_k_mins_global
	init_global_final_NN_result(query_points_num, this->K);
	int DIM = query_points_mat.ncols();
	//int *candidate_query_points_indexes = new int[query_points_num];

	//cout << "m_data rows:" << this->m_data.nrows() << endl;
	//for all
	std::cout << "\ndo find approximate nodes for all query points....\n";
	//this->do_find_approximate_nodes(query_points_mat);	
	this->appr_leaf_node_indexes.resize(query_points_num);
	this->timer_total_approximate_searching.start_my_timer();
	call_do_find_approximate_nodes( query_points_num,
									query_points_mat.get_matrix_raw_data(),
									this->nodes_num,
									this->simplified_tree.get_matrix_raw_data(),
									this->nodes_centers.get_matrix_raw_data(),
									this->appr_leaf_node_indexes.get_matrix_raw_data(),
									DIM
									);
	this->timer_total_approximate_searching.stop_my_timer();
	timer_total_approximate_searching.print("total_approximate_searching time: ");
	//print query points approximate leaf node index
	/*
	std::cout << "query points approximate leaf node indexes:";
    int* matrix_appr_leaf_node_indexs = this->appr_leaf_node_indexes.get_matrix_raw_data();
	for (int i = 0; i < 100; i++){
		std::cout << matrix_appr_leaf_node_indexs[i] << " ";
	}
	std::cout << endl;
	*/
}

/*
This is virtual procedure, a entry for specific task in subclass.
it will be call in init_process_query_points defined in  superclass 'Convex_tree'
*/
template <typename T>
void CUDA_Convex_Tree<T>::new_init_kNN_result(Matrix<T>& query_points_mat,
	                                          FLOAT_TYPE* dist_square_k_mins_global, 
											  int* idx_k_mins_global)
{
	//init final result
	int query_points_num = query_points_mat.nrows();
	//init dist_square_k_mins_global and idx_k_mins_global
	init_global_final_NN_result(query_points_num, this->K);
	
	this->dist_square_k_mins_global = dist_square_k_mins_global;
	this->idx_k_mins_global = idx_k_mins_global;
	
}

/*
This is virtual procedure, a entry for specific task in subclass.
it will be call in init_process_query_points defined in  superclass 'Convex_tree'
*/
template <typename T>
void CUDA_Convex_Tree<T>::init_kNN(FLOAT_TYPE* dist_square_k_mins_global, int* idx_k_mins_global){
	this->dist_square_k_mins_global = dist_square_k_mins_global;
	this->idx_k_mins_global = idx_k_mins_global;
}