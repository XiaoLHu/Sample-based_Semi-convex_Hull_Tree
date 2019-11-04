
#define USE_DOUBLE 0
#if USE_DOUBLE > 0
	#pragma OPENCL EXTENSION cl_khr_fp64: enable
	#define FLOAT_TYPE double
	#define FLOAT_TYPE4 double4
	#define MAX_FLOAT_TYPE      1.7976931348623158e+308
	#define MIN_FLOAT_TYPE     -1.7976931348623158e+308
#else
	#define FLOAT_TYPE float
	#define FLOAT_TYPE4 float4
	#define FLOAT_TYPE8 float8
	#define MAX_FLOAT_TYPE      3.402823466e+38f
	#define MIN_FLOAT_TYPE     -3.402823466e+38f
#endif
#define VECSIZE 4
#define VECSIZE_8 8
#define MAX_KNN 30

// System includes
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <assert.h>
#include <time.h>

// CUDA runtime
#include <cuda_runtime.h>
//Helper functions and utilities to work with CUDA
#include <helper_functions.h>
#include <helper_cuda.h>
#include <device_launch_parameters.h>

typedef struct Convex_Node {
	bool isLeaf;
	int  node_index;
	int  parent_index;
	int  leaf_index;        //the leaf index of this node in all leaf nodes
	int  left_node; 
	int  right_node;
} CONVEX_TREE;

/*----------pulic parameter used in __global__ functions-------------------*/
	int			*d_candidate_query_points_indexes = NULL;	
	FLOAT_TYPE  *d_candidate_query_points_set=NULL;
	int			*d_candidate_query_points_appr_leaf_node_indexes = NULL;	
	FLOAT_TYPE  *d_all_sorted_data_set=NULL;
	int			*d_sorted_data_set_indexes = NULL;	
	CONVEX_TREE *d_tree_struct = NULL;	
	FLOAT_TYPE  *d_all_leaf_nodes_ALPHA_set=NULL;
	FLOAT_TYPE  *d_all_leaf_nodes_BETA_set=NULL;
	int			*d_all_constrains_num_of_each_leaf_nodes=NULL;
	int			*d_all_leaf_nodes_offsets_in_all_ALPHA=NULL;
	int			*d_all_leaf_nodes_ancestor_nodes_ids=NULL;
	int			*d_leaf_nodes_start_pos_in_sorted_data_set=NULL;
	int			*d_pts_num_in_sorted_leaf_nodes=NULL;
	FLOAT_TYPE  *d_dist_k_mins_global_tmp=NULL;
	int			*d_idx_k_mins_global_tmp=NULL;
	long		*d_dist_computation_times_arr=NULL;			
	long		*d_quadprog_times_arr=NULL;
	long		*d_dist_computation_times_in_quadprog=NULL;
	FLOAT_TYPE  *d_nodes_centers=NULL;
/*----------pulic parameter used in __global__ functions-------------------*/

//free memory malloced in CUDA
void free_cuda_mem(){
	if (d_candidate_query_points_indexes != NULL){
	    cudaFree(d_candidate_query_points_indexes);
		d_candidate_query_points_indexes=NULL;
	}
	if (d_candidate_query_points_appr_leaf_node_indexes != NULL){
	    cudaFree(d_candidate_query_points_appr_leaf_node_indexes);
		d_candidate_query_points_appr_leaf_node_indexes=NULL;
	}
	if (d_all_sorted_data_set != NULL){
	    cudaFree(d_all_sorted_data_set);
		d_all_sorted_data_set=NULL;
	}
	if (d_sorted_data_set_indexes != NULL){
	    cudaFree(d_sorted_data_set_indexes);
		d_sorted_data_set_indexes = NULL;	
	}

	if (d_tree_struct != NULL){
	    cudaFree(d_tree_struct);
		d_tree_struct = NULL;	
	}

	if (d_all_leaf_nodes_ALPHA_set != NULL){
	    cudaFree(d_all_leaf_nodes_ALPHA_set);
		d_all_leaf_nodes_ALPHA_set = NULL;	
	}


	if (d_all_leaf_nodes_BETA_set != NULL){
	    cudaFree(d_all_leaf_nodes_BETA_set);
		d_all_leaf_nodes_BETA_set = NULL;	
	}

	if (d_all_constrains_num_of_each_leaf_nodes != NULL){
	    cudaFree(d_all_constrains_num_of_each_leaf_nodes);
		d_all_constrains_num_of_each_leaf_nodes = NULL;	
	}


	if (d_all_leaf_nodes_offsets_in_all_ALPHA != NULL){
	    cudaFree(d_all_leaf_nodes_offsets_in_all_ALPHA);
		d_all_leaf_nodes_offsets_in_all_ALPHA = NULL;	
	}


	if (d_all_leaf_nodes_ancestor_nodes_ids != NULL){
	    cudaFree(d_all_leaf_nodes_ancestor_nodes_ids);
		d_all_leaf_nodes_ancestor_nodes_ids = NULL;	
	}


	if (d_leaf_nodes_start_pos_in_sorted_data_set != NULL){
	    cudaFree(d_leaf_nodes_start_pos_in_sorted_data_set);
		d_leaf_nodes_start_pos_in_sorted_data_set = NULL;	
	}


	if (d_pts_num_in_sorted_leaf_nodes != NULL){
	    cudaFree(d_pts_num_in_sorted_leaf_nodes);
		d_pts_num_in_sorted_leaf_nodes = NULL;	
	}


	if (d_dist_k_mins_global_tmp != NULL){
	    cudaFree(d_dist_k_mins_global_tmp);
		d_dist_k_mins_global_tmp = NULL;	
	}

	if (d_idx_k_mins_global_tmp != NULL){
	    cudaFree(d_idx_k_mins_global_tmp);
		d_idx_k_mins_global_tmp = NULL;	
	}


	if (d_dist_computation_times_arr != NULL){
	    cudaFree(d_dist_computation_times_arr);
		d_dist_computation_times_arr = NULL;	
	}

	
	if (d_quadprog_times_arr != NULL){
	    cudaFree(d_quadprog_times_arr);
		d_quadprog_times_arr = NULL;	
	}		


	if (d_dist_computation_times_in_quadprog != NULL){
	    cudaFree(d_dist_computation_times_in_quadprog);
		d_dist_computation_times_in_quadprog = NULL;	
	}


	if (d_nodes_centers != NULL){
	    cudaFree(d_nodes_centers);
		d_nodes_centers = NULL;	
	}
}

//the inner product of q and p
__device__ FLOAT_TYPE scalar_product_cuda(FLOAT_TYPE* p, FLOAT_TYPE* q, int DIM){
	//DIM will be written in "make_kernel_from_file"
	FLOAT_TYPE result = 0;
	for (int i = 0; i<DIM; i++){
		result += p[i] * q[i];
	}
	return result;
}

__device__ FLOAT_TYPE float_dist_squre_cuda(FLOAT_TYPE *p, FLOAT_TYPE *q, int DIM){
	FLOAT_TYPE dist_tmp=0, tmp=0;
	for (int j = 0; j<DIM; j++){
		tmp = (q[j] - p[j]);
		dist_tmp += tmp*tmp;
	}
	return dist_tmp;
}

__device__ FLOAT_TYPE Compute_distance(FLOAT_TYPE *p1, FLOAT_TYPE *p2, int DIM){
	FLOAT_TYPE sum = 0;
	FLOAT_TYPE *end = p1 + DIM;
	for (; p1 != end; p1++, p2++){
		FLOAT_TYPE d1 = *p1 - *p2;
		d1 *= d1;
		sum = sum + d1;
	}
	return sqrt(sum);
}

//retrun a approximate min dist from q to this convex node
//d_min is still approximate distance, it can be improved or optimized.
//idea: if a point q is outside of this node, then max distance from
//      q to each active constrains (hyperplanes) is the approximate
//      distance. Because the \forall active hyperplane h, we have
//      d_min >= dist(q,h);
__device__ FLOAT_TYPE approximate_min_dist_by_hyper_plane_cuda( FLOAT_TYPE* query_point,
																FLOAT_TYPE* ALPHA,
																FLOAT_TYPE* BETA,
																	   int  ALPPHA_size,
																	   int  DIM){
	FLOAT_TYPE result = 0;
	FLOAT_TYPE tmp_val = 0;
	for (int i = 0; i<ALPPHA_size; i++)
	{
		//DIM will be written in "make_kernel_from_file"
		FLOAT_TYPE* alpha = ALPHA + i*DIM;
		FLOAT_TYPE beta = BETA[i];
		tmp_val = scalar_product_cuda(alpha, query_point,DIM);
		// if there exists a alpha and beta such that alpha[i]'* point +beta[i]<0
		// point is not in the node
		if (tmp_val<0){
			if (result < -tmp_val){
				result = -tmp_val;
			}
		}
	}
	return result;
}


//return true if the d_min from q to this node is larger than dist_compare.
//@param dist_compute_times_in_appr_quadprog: count the dist computation times here
__device__ bool is_appr_min_dist_from_q_larger_by_hyper_plane_cuda( FLOAT_TYPE  *query_point,
																	FLOAT_TYPE  *ALPHA,
																	FLOAT_TYPE  *BETA,
																	int          ALPPHA_size,
																	FLOAT_TYPE   dist_compare,
																	long        *dist_compute_times_in_appr_quadprog,
																	FLOAT_TYPE  *query_point_scalar_product_from_all_nodes,
																	int         *cur_ancestor_nodes_ids,
																	int DIM
																	)
{
	bool result = false;
	int tmp_times = 0;
	int cur_ancestor_node_id = 0;
	for (int i = 0; i<ALPPHA_size; i++){
		FLOAT_TYPE  tmp_dist = BETA[i];
		//---ORIGINAL SCALAR PRODUCT, MANY DUPLICATION, but, in low dim it is faster.
		for (int j = 0; j<DIM; j++){
			tmp_dist += ALPHA[i*DIM + j] * query_point[j];
		}
		tmp_times++;		

		if (tmp_dist<0){
			if (dist_compare <= (tmp_dist*tmp_dist)){
				//if there exists one such hyper plane then return.
				result = true;
				break;
			}
		}

	}
	*dist_compute_times_in_appr_quadprog += tmp_times;
	return result;
}


//brute force computing and update dist_k_mins_private_tmp and idx_k_mins_global_tmp
//pts_num: the number of points in all_sorted_data_set.
__device__ void do_brute_force_and_update_private_cuda(
														FLOAT_TYPE  *cur_query_point,
														int         cur_query_point_index,
														int         pts_num,
														int         cur_leaf_node_start_pos,
														FLOAT_TYPE  *all_sorted_data_set,
															   int  *sorted_data_set_indexes,
														FLOAT_TYPE  *dist_k_mins_private_tmp,
														int         *idx_k_mins_private_tmp,
														int         K_NN,
														int         DIM)
{
	FLOAT_TYPE dist_squre_tmp = 0;
	FLOAT_TYPE tmp = 0;
	int tmp_idx = 0;

	for (int i = 0; i<pts_num; i++){
		dist_squre_tmp = float_dist_squre_cuda(all_sorted_data_set + (cur_leaf_node_start_pos + i)*DIM, cur_query_point,DIM);

		//get the current k^th min_dist_square of current query point
		FLOAT_TYPE cur_k_min_dist_square = dist_k_mins_private_tmp[K_NN - 1];

		if (cur_k_min_dist_square> dist_squre_tmp){
			//printf("update dist_k_mins_private_tmp...\n");
			//printf("cur_k_min_dist_square=%f,  dist_squre_tmp=%f \n",cur_k_min_dist_square,dist_squre_tmp );
			int j = K_NN - 1;
			dist_k_mins_private_tmp[j] = dist_squre_tmp;
			int pts_idx = sorted_data_set_indexes[cur_leaf_node_start_pos + i];
			idx_k_mins_private_tmp[j] = pts_idx;
			for (; j>0; j--){
				if (dist_k_mins_private_tmp[j - 1] > dist_k_mins_private_tmp[j]){
					//printf("new nn found, swap...");
					tmp = dist_k_mins_private_tmp[j - 1];
					dist_k_mins_private_tmp[j - 1] = dist_k_mins_private_tmp[j];
					dist_k_mins_private_tmp[j] = tmp;

					//swap indices
					tmp_idx = idx_k_mins_private_tmp[j - 1];
					idx_k_mins_private_tmp[j - 1] = idx_k_mins_private_tmp[j];
					idx_k_mins_private_tmp[j] = tmp_idx;
				}
				else break;
			}
		}
	}
}

//brute force computing and update dist_k_mins_private_tmp and idx_k_mins_global_tmp
//pts_num: the number of points in all_sorted_data_set.
__device__ void new_do_brute_force_and_update_private_cuda(
	FLOAT_TYPE  *cur_query_point,
	int         cur_query_point_index,
	int         pts_num,
	int         cur_leaf_node_start_pos,
	FLOAT_TYPE  *all_sorted_data_set,
	int         *sorted_data_set_indexes,
	FLOAT_TYPE  *dist_k_mins_private_tmp,
	int         *idx_k_mins_private_tmp,
	int         *remain_index,
	int         K_NN,
	int         DIM)
{
	FLOAT_TYPE dist_squre_tmp = 0;
	FLOAT_TYPE tmp = 0;
	int tmp_idx = 0;

	for (int i = 0; i<pts_num; i++){
		dist_squre_tmp = float_dist_squre_cuda(all_sorted_data_set + (cur_leaf_node_start_pos + i)*DIM, cur_query_point, DIM);

		//get the current k^th min_dist_square of current query point
		FLOAT_TYPE cur_k_min_dist_square = dist_k_mins_private_tmp[K_NN - 1];

		if (cur_k_min_dist_square> dist_squre_tmp){
			//printf("update dist_k_mins_private_tmp...\n");
			//printf("cur_k_min_dist_square=%f,  dist_squre_tmp=%f \n",cur_k_min_dist_square,dist_squre_tmp );
			int j = K_NN - 1;
			dist_k_mins_private_tmp[j] = dist_squre_tmp;
			int pts_idx = sorted_data_set_indexes[cur_leaf_node_start_pos + i];
			idx_k_mins_private_tmp[j] = remain_index[pts_idx];
			for (; j>0; j--){
				if (dist_k_mins_private_tmp[j - 1] > dist_k_mins_private_tmp[j]){
					//printf("new nn found, swap...");
					tmp = dist_k_mins_private_tmp[j - 1];
					dist_k_mins_private_tmp[j - 1] = dist_k_mins_private_tmp[j];
					dist_k_mins_private_tmp[j] = tmp;

					//swap indices
					tmp_idx = idx_k_mins_private_tmp[j - 1];
					idx_k_mins_private_tmp[j - 1] = idx_k_mins_private_tmp[j];
					idx_k_mins_private_tmp[j] = tmp_idx;
				}
				else break;
			}
		}
	}
}


/*
0 candidate_query_points_num        : the number of current candidate query points, in the case of all query points set
                                      is too large, we can submit subset of query sets to this kernel.
1 candidate_query_points_indexes    : the indexes of current query points in all query points set
2 candidate_query_points_set        : the current query points data set
3 candidate_query_points_appr_leaf_node_indexes : the approximate leaf node for candidate query points
4 all_sorted_data_set               : all sorted data
5 sorted_data_set_indexes           : all points indexes in sorted data set
6 tree_struct                       : the tree structure of the whole tree. It is not used now.
7 all_leaf_nodes_ALPHA_set          : ALPHA set of all leaf nodes
8 leaf_nodes_BETA_set               : BETA set of all leaf nodes
9 all_constrains_num_of_each_leaf_nodes    : all_constrains_num_of_each_nodes[i]=j means i^th leaf nodes has j constrains, i.e. has j alphas and betas
10 all_leaf_nodes_offsets_in_all_ALPHA     : the offset of each leaf node in ALPHA
11 leaf_node_num                           : the number of leaf nodes
12 all_leaf_nodes_ancestor_nodes_ids       : the ancestor nodes ids of each leaf nodes
13 leaf_nodes_start_pos_in_sorted_data_set : specify the start position from which each sorted leave node in sorted data set
14 pts_num_in_sorted_leaf_nodes      : the length of points saved in each sorted leaf node
15 dist_k_mins_global_tmp            : the K min-distance of all query points, the length of dist_mins_global_tmp is K* query_points_size
16 idx_k_mins_global_tmp             : the indexes of the K nearest neighbors, the length of dist_mins_global_tmp is K* query_points_size
17 K_NN                              : the value of K
18 dist_computation_times_arr        : dist_computation_times_arr[i] saves total distance computation times of the i^th point and
19 quadprog_times_arr                : quadprog_times_arr[i] approximate quadprog times of the i^th point.
20 dist_computation_times_in_quadprog: dist_computation_times_in_quadprog[i] saves the total distance computation times
                                       in quadprog of the i^th point.
*/
__global__ void do_finding_KNN_by_leaf_order_cuda(
												int          candidate_query_points_num,
												int         *candidate_query_points_indexes,
										 FLOAT_TYPE         *candidate_query_points_set,
												int         *candidate_query_points_appr_leaf_node_indexes,
										 FLOAT_TYPE         *all_sorted_data_set,
												int         *sorted_data_set_indexes,
										CONVEX_TREE         *tree_struct,
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
												int          DIM,
												int          loop_id)
{
	//---global thread id
	//int tid = blockIdx.x;
	//int tid = blockDim.x * blockIdx.x + threadIdx.x;
	//int tid = (blockIdx.y * gridDim.x + blockIdx.x) * (blockDim.x * blockDim.y) + threadIdx.y * blockDim.y + threadIdx.x;
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int tid = x + y * blockDim.x * gridDim.x;
	//printf(" [loopid=%d, tid_ori=%d], ", loop_id,tid);   
	//tid +=loop_id* blocks_per_time;

	if (tid >= candidate_query_points_num){
		return;
	}
	
	//---count the distance computation times in approximate quadprog.
	long cur_dist_compute_times_in_appr_quadprog = 0;

	int cur_query_points_appr_leaf_node_indexes = candidate_query_points_appr_leaf_node_indexes[tid];

	int cur_leaf_node_start_pos = leaf_nodes_start_pos_in_sorted_data_set[cur_query_points_appr_leaf_node_indexes];

	/*---------------------------------------------------------------------------------------------------------------
	//---query_points_nodes_alpha_scalar_product is not used in is_appr_min_dist_from_q_larger_by_hyper_plane now.
	//---because visiting  query_points_nodes_alpha_scalar_product randomly seems slow.
	//---private scalar product between current query point and  all ALPHAs, which are all initialized to 0.
	//---each node has a alpha constraint, a well as constraints of its ancestors nodes.
	//---'ALL_NODES_NUM' will be written before kernel is created.
	----------------------------------------------------------------------------------------------------------------*/\
	
	/*-----------------Copy global data as local data: visiting global data is relative slow in devices-----------------------------*/
	int quadprog_times_private = 0;
	for (int j = 0; j < K_NN; j++){
		dist_k_mins_global_tmp[candidate_query_points_indexes[tid]*K_NN + j] = MAX_FLOAT_TYPE;
	 	idx_k_mins_global_tmp[candidate_query_points_indexes[tid]*K_NN + j] = -1;
	}
	//---here is tid instead of cur_query_point_index, tid is the offset of current query point in candidate_query_points_set
	FLOAT_TYPE* cur_query_point = candidate_query_points_set + tid*DIM;

	/*-----------------------------------------------------------------------------------------------------------------------------*/

	long dist_computation_times_tmp = 0;
	int pts_num = pts_num_in_sorted_leaf_nodes[cur_query_points_appr_leaf_node_indexes];

	//---find approximate kNN in its approximate nodes.

	do_brute_force_and_update_private_cuda( cur_query_point, candidate_query_points_indexes[tid],
											pts_num, cur_leaf_node_start_pos,
											all_sorted_data_set, sorted_data_set_indexes,
											dist_k_mins_global_tmp+candidate_query_points_indexes[tid]*K_NN, 
											idx_k_mins_global_tmp+candidate_query_points_indexes[tid]*K_NN,
											K_NN, DIM);

	//---add distance computation times
	//dist_computation_times_tmp += pts_num;
	
	for (int i = 0; i < leaf_node_num; i++) {
		if (i == cur_query_points_appr_leaf_node_indexes)
			continue;
		int alpha_offset = all_leaf_nodes_offsets_in_all_ALPHA[i];
		int constrains_num = all_constrains_num_of_each_leaf_nodes[i];

		//---get the current k^th min_dist_square of current query point
		FLOAT_TYPE cur_k_min_dist_square = dist_k_mins_global_tmp[candidate_query_points_indexes[tid]*K_NN+K_NN - 1];

		FLOAT_TYPE* cur_ALPHAT = all_leaf_nodes_ALPHA_set + alpha_offset*DIM;
		FLOAT_TYPE* cur_BETA = all_leaf_nodes_BETA_set + alpha_offset;

		//---the number of ancestor nodes is the same as the size of constraints
		int* cur_ancestor_nodes_ids = all_leaf_nodes_ancestor_nodes_ids + alpha_offset;

		//---check whether the current node is a candidate for current query point
		if (!is_appr_min_dist_from_q_larger_by_hyper_plane_cuda(cur_query_point, cur_ALPHAT, cur_BETA,
																constrains_num, cur_k_min_dist_square,
																&cur_dist_compute_times_in_appr_quadprog,
																NULL,
																cur_ancestor_nodes_ids,
																DIM
																) )
		{
			//---do brute force distance computation here, and update dist_k_mins_global_tmp and idx_k_mins_global_tmp
			//---get the number of points saved in current node
			//---i is cur leaf node index, not leaf node ori_index
			int pts_num = pts_num_in_sorted_leaf_nodes[i];
			int cur_leaf_node_start_pos = leaf_nodes_start_pos_in_sorted_data_set[i];
			do_brute_force_and_update_private_cuda( cur_query_point, candidate_query_points_indexes[tid],
													pts_num, cur_leaf_node_start_pos,
													all_sorted_data_set, sorted_data_set_indexes,													
													dist_k_mins_global_tmp+candidate_query_points_indexes[tid]*K_NN, 
													idx_k_mins_global_tmp+candidate_query_points_indexes[tid]*K_NN,
													K_NN, DIM);
		}
	}
	
}



/*
0 candidate_query_points_num        : the number of current candidate query points, in the case of all query points set
                                      is too large, we can submit subset of query sets to this kernel.
1 candidate_query_points_indexes    : the indexes of current query points in all query points set
2 candidate_query_points_set        : the current query points data set
  candidate_query_points_appr_leaf_node_indexes : the approximate leaf node for candidate query points
4 all_sorted_data_set               : all sorted data
5 sorted_data_set_indexes           : all points indexes in sorted data set
6 tree_struct                       : the tree structure of the whole tree. It is not used now.
7 all_leaf_nodes_ALPHA_set          : ALPHA set of all leaf nodes
8 leaf_nodes_BETA_set               : BETA set of all leaf nodes
9 all_constrains_num_of_each_leaf_nodes    : all_constrains_num_of_each_nodes[i]=j means i^th leaf nodes has j constrains, i.e. has j alphas and betas
10 all_leaf_nodes_offsets_in_all_ALPHA     : the offset of each leaf node in ALPHA
11 leaf_node_num                           : the number of leaf nodes
12 all_leaf_nodes_ancestor_nodes_ids       : the ancestor nodes ids of each leaf nodes
13 leaf_nodes_start_pos_in_sorted_data_set : specify the start position from which each sorted leave node in sorted data set
14 pts_num_in_sorted_leaf_nodes      : the length of points saved in each sorted leaf node
15 dist_k_mins_global_tmp            : the K min-distance of all query points, the length of dist_mins_global_tmp is K* query_points_size
16 idx_k_mins_global_tmp             : the indexes of the K nearest neighbors, the length of dist_mins_global_tmp is K* query_points_size
17 K_NN                              : the value of K
18 dist_computation_times_arr        : dist_computation_times_arr[i] saves total distance computation times of the i^th point and
19 quadprog_times_arr                : quadprog_times_arr[i] approximate quadprog times of the i^th point.
20 dist_computation_times_in_quadprog: dist_computation_times_in_quadprog[i] saves the total distance computation times
                                       in quadprog of the i^th point.
*/
__global__ void new_do_finding_KNN_by_leaf_order_cuda(
												int          candidate_query_points_num,
												int         *candidate_query_points_indexes,
										 FLOAT_TYPE         *candidate_query_points_set,
										 FLOAT_TYPE         *all_sorted_data_set,
												int         *sorted_data_set_indexes,
										CONVEX_TREE         *tree_struct,
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
												int         *remain_index,
												int          K_NN,
											   long         *dist_computation_times_arr,
											   long         *quadprog_times_arr,
											   long         *dist_computation_times_in_quadprog,
												int          NODES_NUM,
												int          DIM,
												int          loop_id)
{
	//---global thread id
	//int tid = blockIdx.x;
	//int tid = blockDim.x * blockIdx.x + threadIdx.x;
	//int tid = (blockIdx.y * gridDim.x + blockIdx.x) * (blockDim.x * blockDim.y) + threadIdx.y * blockDim.y + threadIdx.x;
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int tid = x + y * blockDim.x * gridDim.x;
	//printf(" [loopid=%d, tid_ori=%d], ", loop_id,tid);   
	//tid +=loop_id* blocks_per_time;

	if (tid >= candidate_query_points_num){
		return;
	}
	
	//---count the distance computation times in approximate quadprog.
	long cur_dist_compute_times_in_appr_quadprog = 0;

	//int cur_query_points_appr_leaf_node_indexes = candidate_query_points_appr_leaf_node_indexes[tid];

	//int cur_leaf_node_start_pos = leaf_nodes_start_pos_in_sorted_data_set[cur_query_points_appr_leaf_node_indexes];

	/*---------------------------------------------------------------------------------------------------------------
	//---query_points_nodes_alpha_scalar_product is not used in is_appr_min_dist_from_q_larger_by_hyper_plane now.
	//---because visiting  query_points_nodes_alpha_scalar_product randomly seems slow.
	//---private scalar product between current query point and  all ALPHAs, which are all initialized to 0.
	//---each node has a alpha constraint, a well as constraints of its ancestors nodes.
	//---'ALL_NODES_NUM' will be written before kernel is created.
	----------------------------------------------------------------------------------------------------------------*/\
	
	/*-----------------Copy global data as local data: visiting global data is relative slow in devices-----------------------------*/
	int quadprog_times_private = 0;
	/*for (int j = 0; j < K_NN; j++){
		dist_k_mins_global_tmp[candidate_query_points_indexes[tid]*K_NN + j] = MAX_FLOAT_TYPE;
	 	idx_k_mins_global_tmp[candidate_query_points_indexes[tid]*K_NN + j] = -1;
	}
	*/
	//---here is tid instead of cur_query_point_index, tid is the offset of current query point in candidate_query_points_set
	FLOAT_TYPE* cur_query_point = candidate_query_points_set + tid*DIM;

	/*-----------------------------------------------------------------------------------------------------------------------------*/

	long dist_computation_times_tmp = 0;
	//int pts_num = pts_num_in_sorted_leaf_nodes[cur_query_points_appr_leaf_node_indexes];

	//---find approximate kNN in its approximate nodes.

	/*do_brute_force_and_update_private_cuda( cur_query_point, candidate_query_points_indexes[tid],
											pts_num, cur_leaf_node_start_pos,
											all_sorted_data_set, sorted_data_set_indexes,
											dist_k_mins_global_tmp+candidate_query_points_indexes[tid]*K_NN, 
											idx_k_mins_global_tmp+candidate_query_points_indexes[tid]*K_NN,
											K_NN, DIM);
											*/

	//---add distance computation times
	//dist_computation_times_tmp += pts_num;

	for (int i = 0; i < leaf_node_num; i++) {
		//if (i == cur_query_points_appr_leaf_node_indexes)
			//continue;
		int alpha_offset = all_leaf_nodes_offsets_in_all_ALPHA[i];
		int constrains_num = all_constrains_num_of_each_leaf_nodes[i];

		//---get the current k^th min_dist_square of current query point
		FLOAT_TYPE cur_k_min_dist_square = dist_k_mins_global_tmp[candidate_query_points_indexes[tid]*K_NN+K_NN - 1];

		FLOAT_TYPE* cur_ALPHAT = all_leaf_nodes_ALPHA_set + alpha_offset*DIM;
		FLOAT_TYPE* cur_BETA = all_leaf_nodes_BETA_set + alpha_offset;

		//---the number of ancestor nodes is the same as the size of constraints
		int* cur_ancestor_nodes_ids = all_leaf_nodes_ancestor_nodes_ids + alpha_offset;

		//---check whether the current node is a candidate for current query point
		if (!is_appr_min_dist_from_q_larger_by_hyper_plane_cuda(cur_query_point, cur_ALPHAT, cur_BETA,
																constrains_num, cur_k_min_dist_square,
																&cur_dist_compute_times_in_appr_quadprog,
																NULL,
																cur_ancestor_nodes_ids,
																DIM
																) )
		{
			//---do brute force distance computation here, and update dist_k_mins_global_tmp and idx_k_mins_global_tmp
			//---get the number of points saved in current node
			//---i is cur leaf node index, not leaf node ori_index
			int pts_num = pts_num_in_sorted_leaf_nodes[i];
			int cur_leaf_node_start_pos = leaf_nodes_start_pos_in_sorted_data_set[i];
			new_do_brute_force_and_update_private_cuda( cur_query_point, candidate_query_points_indexes[tid],
													    pts_num, cur_leaf_node_start_pos,
													    all_sorted_data_set, sorted_data_set_indexes,													
													    dist_k_mins_global_tmp+candidate_query_points_indexes[tid]*K_NN, 
													    idx_k_mins_global_tmp+candidate_query_points_indexes[tid]*K_NN,
													    remain_index,K_NN, DIM);
		}
	}
}


__global__ void print_float_Data(FLOAT_TYPE   *dist_k_mins_global_tmp, int* idx_k_mins_global_tmp, int loop_id){
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	int tid_ori=tid;
	tid+= loop_id*1024;
	int K_NN=30;
	int cur_query_point_index=tid;
	for (int j = 0; j<K_NN; j++){
		printf(" [tid=%d, j=%d,  dist =%f, idx=%d]   ", tid, j, dist_k_mins_global_tmp[cur_query_point_index*K_NN + j], idx_k_mins_global_tmp[cur_query_point_index*K_NN + j]);		
	}
}


/*
//find kNN by brute force
0 data_set                          : the number of current candidate query points
1 data_set_size                     : cardinal
2 query_points                      : all query points
3 query_points_size                 : the length of query_points
4 dist_k_mins_global_tmp            : the K min-distance of all query points,
the length of dist_mins_global_tmp is K* query_points_size
5 idx_k_mins_global_tmp             : the indexes of the K nearest neighbors,
the length of dist_mins_global_tmp is K* query_points_size
6 K_NN                              : the value of K
*/
/*
__global__ void do_brute_force_KNN_cuda(
	FLOAT_TYPE *data_set,int data_set_size,
	FLOAT_TYPE *query_set,int query_set_size,
	FLOAT_TYPE *KNN_index_with_dist,
	int K,int DIM)
{
	//global thread id
	int tid = blockDim.x *blockIdx.x + threadIdx.x;

	if (tid > query_set_size){
		return;
	}

	unsigned int current_query_point_index = tid;
	FLOAT_TYPE *temp = new FLOAT_TYPE[2 * K];
	FLOAT_TYPE *p1, *p2;
	int tmp;
	FLOAT_TYPE d, max_dist, max_idx;
	p1 = query_set + current_query_point_index*DIM;
	for (int i = 0; i < data_set_size; i++){
		p2 = data_set + i*DIM;
		d = Compute_distance(p1, p2, DIM);
		if (i < K){
			temp[i * 2] = i;
			temp[i * 2 + 1] = d;
		}
		else{
			tmp = 0;
			max_idx = temp[0];
			max_dist = temp[1];
			for (int j = 1; j < K; j++){
				if (temp[2 * j + 1] > max_dist){
					tmp = j;
					max_idx = temp[2 * j];
					max_dist = temp[2 * j + 1];
				}
			}
			if (d < max_dist){
				temp[tmp * 2] = i;
				temp[tmp * 2 + 1] = d;
			}
		}
	}
	memcpy(KNN_index_with_dist + current_query_point_index * 2 * K, temp, (2 * K)*sizeof(FLOAT_TYPE));
	//cudaMemcpy(KNN_index_with_dist + current_query_point_index * 2 * K, temp, (2 * K)*sizeof(FLOAT_TYPE), cudaMemcpyDeviceToDevice);
}
*/
__global__  void do_brute_force_KNN_cuda(
						FLOAT_TYPE *data_set,
							   int  data_set_size,
						FLOAT_TYPE *query_points,
							   int  query_points_size,
						FLOAT_TYPE *dist_k_mins_global_tmp,
							   int *idx_k_mins_global_tmp,
							   int  K_NN,
							   int  DIM)
{
	// global thread id
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid >= query_points_size){
		return;
	}

	unsigned int current_query_point_index = tid;

	//---init the distance as MAX_FLOAT_TYPE
	
	for (int i = 0; i<K_NN; i++){
		dist_k_mins_global_tmp[current_query_point_index*K_NN + i] = MAX_FLOAT_TYPE;
	}

	//get the current k^th min_dist_square of current query point
	FLOAT_TYPE cur_k_min_dist_square = dist_k_mins_global_tmp[current_query_point_index*K_NN + K_NN - 1];

	//if (tid==checkid)
	//   printf("cur_k_min_dist_square =%f \n",cur_k_min_dist_square);


	FLOAT_TYPE dist_square_tmp = 0;
	FLOAT_TYPE tmp = 0;
	int tmp_idx = 0;

	//local copy
	FLOAT_TYPE* cur_query_point_private = new FLOAT_TYPE[DIM];
	for (int i = 0; i<DIM; i++){
		cur_query_point_private[i] = query_points[current_query_point_index*DIM + i];
	}


	for (int i = 0; i<data_set_size; i++){
		dist_square_tmp = 0;
		cur_k_min_dist_square = dist_k_mins_global_tmp[current_query_point_index*K_NN + K_NN - 1];
		for (int j = 0; j<DIM; j++){
			tmp = data_set[i*DIM + j] - cur_query_point_private[j];
			dist_square_tmp += tmp*tmp;
			//printf("tmp =%f, dist_square_tmp=%f\n",tmp,dist_square_tmp);
		}
		//printf("dist_square_tmp =%f, cur_k_min_dist_square=%f \n",dist_square_tmp, cur_k_min_dist_square);
		if (cur_k_min_dist_square> dist_square_tmp){
			//printf("update dist_k_mins_global_tmp...\n");
			int j = K_NN - 1;
			dist_k_mins_global_tmp[current_query_point_index*K_NN + j] = dist_square_tmp;
			idx_k_mins_global_tmp[current_query_point_index*K_NN + j] = i;

			for (; j>0; j--){
				if (dist_k_mins_global_tmp[current_query_point_index*K_NN + j - 1] > dist_k_mins_global_tmp[current_query_point_index*K_NN + j]){
					//printf("new nn found, swap...");
					tmp = dist_k_mins_global_tmp[current_query_point_index*K_NN + j - 1];
					dist_k_mins_global_tmp[current_query_point_index*K_NN + j - 1] = dist_k_mins_global_tmp[current_query_point_index*K_NN + j];
					dist_k_mins_global_tmp[current_query_point_index*K_NN + j] = tmp;

					//swap indices
					tmp_idx = idx_k_mins_global_tmp[current_query_point_index*K_NN + j - 1];
					idx_k_mins_global_tmp[current_query_point_index*K_NN + j - 1] = idx_k_mins_global_tmp[current_query_point_index*K_NN + j];
					idx_k_mins_global_tmp[current_query_point_index*K_NN + j] = tmp_idx;
				}
				else break;
			}
		}
	}
}

/*
//find kNN by brute force for outliers
0 data_set                          : the number of current candidate query points
1 data_set_size                     : cardinal
2 query_points                      : all query points
3 query_points_size                 : the length of query_points
4 dist_k_mins_global_tmp            : the K min-distance of all query points,
the length of dist_mins_global_tmp is K* query_points_size
5 idx_k_mins_global_tmp             : the indexes of the K nearest neighbors,
the length of dist_mins_global_tmp is K* query_points_size
6 K_NN                              : the value of K
*/
__global__  void do_brute_force_KNN_for_outliers_cuda(
	FLOAT_TYPE          *data_set,
	int                 data_set_size,
	int                 *data_indexes,
	FLOAT_TYPE          *query_points,
	int                 query_points_size,
	FLOAT_TYPE          *dist_k_mins_global_tmp,
	int                 *idx_k_mins_global_tmp,
	int                 K_NN,
	int                 DIM)
{
	// global thread id
	int tid = blockDim.x * blockIdx.x + threadIdx.x;

	//int tid = threadIdx.x;
	//int tid = 0;
	//printf("tid =%d \n",tid);

	if (tid >= query_points_size){
		return;
	}

	//printf("tid=%d, data_set_size =%d,query_points_size=%d  \n",tid,data_set_size,query_points_size);


	unsigned int current_query_point_index = tid;

	//get the current k^th min_dist_square of current query point
	FLOAT_TYPE cur_k_min_dist_square = dist_k_mins_global_tmp[current_query_point_index*K_NN + K_NN - 1];

	//if (tid==checkid)
	//   printf("cur_k_min_dist_square =%f \n",cur_k_min_dist_square);


	FLOAT_TYPE dist_square_tmp = 0;
	FLOAT_TYPE tmp = 0;
	int tmp_idx = 0;

	//local copy
	FLOAT_TYPE* cur_query_point_private = new FLOAT_TYPE[DIM];
	for (int i = 0; i<DIM; i++){
		cur_query_point_private[i] = query_points[current_query_point_index*DIM + i];
	}


	for (int i = 0; i < data_set_size; i++){
		dist_square_tmp = 0;
		cur_k_min_dist_square = dist_k_mins_global_tmp[current_query_point_index*K_NN + K_NN - 1];
		for (int j = 0; j<DIM; j++){
			tmp = data_set[i*DIM + j] - cur_query_point_private[j];
			dist_square_tmp += tmp*tmp;
			//printf("tmp =%f, dist_square_tmp=%f\n",tmp,dist_square_tmp);
		}
		//printf("dist_square_tmp =%f, cur_k_min_dist_square=%f \n",dist_square_tmp, cur_k_min_dist_square);
		if (cur_k_min_dist_square> dist_square_tmp){
			//printf("update dist_k_mins_global_tmp...\n");
			int j = K_NN - 1;
			dist_k_mins_global_tmp[current_query_point_index*K_NN + j] = dist_square_tmp;
			idx_k_mins_global_tmp[current_query_point_index*K_NN + j] = data_indexes[i];

			for (; j>0; j--){
				if (dist_k_mins_global_tmp[current_query_point_index*K_NN + j - 1] > dist_k_mins_global_tmp[current_query_point_index*K_NN + j]){
					//printf("new nn found, swap...");
					tmp = dist_k_mins_global_tmp[current_query_point_index*K_NN + j - 1];
					dist_k_mins_global_tmp[current_query_point_index*K_NN + j - 1] = dist_k_mins_global_tmp[current_query_point_index*K_NN + j];
					dist_k_mins_global_tmp[current_query_point_index*K_NN + j] = tmp;

					//swap indices
					tmp_idx = idx_k_mins_global_tmp[current_query_point_index*K_NN + j - 1];
					idx_k_mins_global_tmp[current_query_point_index*K_NN + j - 1] = idx_k_mins_global_tmp[current_query_point_index*K_NN + j];
					idx_k_mins_global_tmp[current_query_point_index*K_NN + j] = tmp_idx;
				}
				else break;
			}
		}
	}
}

__device__ int get_min_k_index_cuda(FLOAT_TYPE  *dist_k_mins_private_tmp, int K_NN){
	FLOAT_TYPE tmp_max = dist_k_mins_private_tmp[0];
	int result = 0;
	for (int i = 1; i<K_NN; i++){
		if (dist_k_mins_private_tmp[i]>tmp_max){
			result = i;
			tmp_max = dist_k_mins_private_tmp[i];
		}
	}
	return result;
}


bool init_CUDA_device(){
	int count;
	cudaGetDeviceCount(&count);
	//printf("\nDevice number: %d\n", count);
	if (count == 0) {
		fprintf(stderr, "There is no device.\n");
		return false;
	}
	int i;
	for (i = 0; i < count; i++) {
		cudaDeviceProp prop;
		if (cudaGetDeviceProperties(&prop, i) == cudaSuccess) {
			//printf("Device name :%s\n", prop.name);
			//printf("Device major :%d\n", prop.major);
			//printf("Device multiProcessorCount :%d\n", prop.multiProcessorCount);
			//printf("Device maxThreadsPerBlock :%d\n", prop.maxThreadsPerBlock);
			//printf("Device totalGlobalMem :%ld\n", prop.totalGlobalMem);
			//printf("Device totalConstMem :%ld\n", prop.totalConstMem);
			if (prop.major >= 1) {
				break;
			}
		}
	}
	if (i == count) {
		fprintf(stderr, "There is no device supporting CUDA 1.x.\n");
		return false;
	}
	cudaSetDevice(i);
	return true;

}


/**
 * Host main routine
 */
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
				int          DIM)
{
	clock_t start, finish,start1, finish1;
	float duration ;
	bool cuda_init =init_CUDA_device();
	if (cuda_init){
	    printf("\nsucced for initializing CUDA");
	}
	cudaError_t err = cudaSuccess;

	//Launch the Vector Add CUDA Kernel
    printf("\nCUDA Malloc Memory.....\n ");

	start=clock();
	size_t size = candidate_query_points_num * sizeof(int);
	if (d_candidate_query_points_indexes==NULL){
		err = cudaMalloc((void **)&d_candidate_query_points_indexes, size);
		err = cudaMemcpy(d_candidate_query_points_indexes, candidate_query_points_indexes, size, cudaMemcpyHostToDevice);
	}

	size = candidate_query_points_num * sizeof(FLOAT_TYPE)*DIM;
	if (d_candidate_query_points_set==NULL){
		err = cudaMalloc((void **)&d_candidate_query_points_set, size);
		err = cudaMemcpy(d_candidate_query_points_set, candidate_query_points_set, size, cudaMemcpyHostToDevice);
	}

	size = candidate_query_points_num * sizeof(int);
	if (d_candidate_query_points_appr_leaf_node_indexes==NULL){
		err = cudaMalloc((void **)&d_candidate_query_points_appr_leaf_node_indexes, size);
		err = cudaMemcpy(d_candidate_query_points_appr_leaf_node_indexes, candidate_query_points_appr_leaf_node_indexes, size, cudaMemcpyHostToDevice);
	}


	size = sorted_data_len * sizeof(FLOAT_TYPE)*DIM;
	if (d_all_sorted_data_set==NULL){
		err = cudaMalloc((void **)&d_all_sorted_data_set, size);
		err = cudaMemcpy(d_all_sorted_data_set, all_sorted_data_set, size, cudaMemcpyHostToDevice);
	}


	size = sorted_data_len * sizeof(int);
	if (d_sorted_data_set_indexes==NULL){
		err = cudaMalloc((void **)&d_sorted_data_set_indexes, size);
		err = cudaMemcpy(d_sorted_data_set_indexes, sorted_data_set_indexes, size, cudaMemcpyHostToDevice);
	}

	size=tree_nodes_num*sizeof(CONVEX_TREE);
	if (d_tree_struct==NULL){
		err = cudaMalloc((void **)&d_tree_struct, size);
		err = cudaMemcpy(d_tree_struct, tree_struct, size, cudaMemcpyHostToDevice);
	}

	size= all_leaf_nodes_constraint_num* sizeof(FLOAT_TYPE)*DIM;
	if (d_all_leaf_nodes_ALPHA_set==NULL){
		err = cudaMalloc((void **)&d_all_leaf_nodes_ALPHA_set, size);
		err = cudaMemcpy(d_all_leaf_nodes_ALPHA_set, all_leaf_nodes_ALPHA_set, size, cudaMemcpyHostToDevice);
	}

	size= all_leaf_nodes_constraint_num* sizeof(FLOAT_TYPE);
	if (d_all_leaf_nodes_BETA_set==NULL){
		err = cudaMalloc((void **)&d_all_leaf_nodes_BETA_set, size);
		err = cudaMemcpy(d_all_leaf_nodes_BETA_set, all_leaf_nodes_BETA_set, size, cudaMemcpyHostToDevice);
	}

	size= leaf_node_num*sizeof(int);
	if (d_all_constrains_num_of_each_leaf_nodes==NULL){
		err = cudaMalloc((void **)&d_all_constrains_num_of_each_leaf_nodes, size);
		err = cudaMemcpy(d_all_constrains_num_of_each_leaf_nodes, all_constrains_num_of_each_leaf_nodes, size, cudaMemcpyHostToDevice);
	}
	
	if (d_all_leaf_nodes_offsets_in_all_ALPHA==NULL){
		err = cudaMalloc((void **)&d_all_leaf_nodes_offsets_in_all_ALPHA, size);
		err = cudaMemcpy(d_all_leaf_nodes_offsets_in_all_ALPHA, all_leaf_nodes_offsets_in_all_ALPHA, size, cudaMemcpyHostToDevice);
	}

	if (d_all_leaf_nodes_ancestor_nodes_ids==NULL){
		err = cudaMalloc((void **)&d_all_leaf_nodes_ancestor_nodes_ids, size);
		err = cudaMemcpy(d_all_leaf_nodes_ancestor_nodes_ids, all_leaf_nodes_ancestor_nodes_ids, size, cudaMemcpyHostToDevice);
	}
	
	if (d_leaf_nodes_start_pos_in_sorted_data_set==NULL){
		err = cudaMalloc((void **)&d_leaf_nodes_start_pos_in_sorted_data_set, size);
		err = cudaMemcpy(d_leaf_nodes_start_pos_in_sorted_data_set, leaf_nodes_start_pos_in_sorted_data_set, size, cudaMemcpyHostToDevice);
	}
	
	if (d_pts_num_in_sorted_leaf_nodes==NULL){
		err = cudaMalloc((void **)&d_pts_num_in_sorted_leaf_nodes, size);
		err = cudaMemcpy(d_pts_num_in_sorted_leaf_nodes, pts_num_in_sorted_leaf_nodes, size, cudaMemcpyHostToDevice);
	}


	size= candidate_query_points_num* sizeof(FLOAT_TYPE)* K_NN;
	if (d_dist_k_mins_global_tmp==NULL){
		err = cudaMalloc((void **)&d_dist_k_mins_global_tmp, size);
	}

	size= candidate_query_points_num* sizeof(int)* K_NN;
	if (d_idx_k_mins_global_tmp==NULL){
		err = cudaMalloc((void **)&d_idx_k_mins_global_tmp, size);
	}

	size= candidate_query_points_num*sizeof(long);
	err = cudaMalloc((void **)&d_dist_computation_times_arr, size);
	
	err = cudaMalloc((void **)&d_quadprog_times_arr, size);
			
	err = cudaMalloc((void **)&d_dist_computation_times_in_quadprog, size);
	int task_per_num=100000/256;

	finish=clock();
	duration = (float)(finish-start)/ CLOCKS_PER_SEC;
	//printf( "\n CUDA Malloc Memory Time %f\n", duration);

	//printf ("\n calling do_finding_KNN_by_leaf_order_cuda.....\n");
	int block_num = candidate_query_points_num / 1024 + 1;
	dim3  blocks(block_num,1), threads(1024,1); 
	//int blocks_per_time=6*16*128*8;
	
	start1 = clock();
	do_finding_KNN_by_leaf_order_cuda <<< blocks, threads>>>(
				candidate_query_points_num,
				d_candidate_query_points_indexes,
				d_candidate_query_points_set,
				d_candidate_query_points_appr_leaf_node_indexes,
				d_all_sorted_data_set,
				d_sorted_data_set_indexes,
				d_tree_struct,
				d_all_leaf_nodes_ALPHA_set,
				d_all_leaf_nodes_BETA_set,
				d_all_constrains_num_of_each_leaf_nodes,
				d_all_leaf_nodes_offsets_in_all_ALPHA,
				leaf_node_num,
				d_all_leaf_nodes_ancestor_nodes_ids,
				d_leaf_nodes_start_pos_in_sorted_data_set,
				d_pts_num_in_sorted_leaf_nodes,
				d_dist_k_mins_global_tmp,
				d_idx_k_mins_global_tmp,
				K_NN,
				d_dist_computation_times_arr,
				d_quadprog_times_arr,
				d_dist_computation_times_in_quadprog,
				NODES_NUM,
				DIM,
				1);
	
    err = cudaGetLastError();
	if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
        //exit(EXIT_FAILURE);
    }
	finish1 = clock();
	duration = (float)(finish1-start1)/ CLOCKS_PER_SEC;
	//printf( "\n performing do_finding_KNN_by_leaf_order_cuda time %f milsec %f s\n",(float)(finish1-start1), duration);

	//printf( "----print device matrix-------");
	//print_float_Data <<<4, 256>>>(d_dist_k_mins_global_tmp,d_idx_k_mins_global_tmp,0);


    // Copy the device result vector in device memory to the host result vector
    // in host memory.

	//printf("\n copy data from GPU....\n");
	size= candidate_query_points_num* sizeof(FLOAT_TYPE)* K_NN;
	err = cudaMemcpy(dist_k_mins_global_tmp, d_dist_k_mins_global_tmp, size, cudaMemcpyDeviceToHost);

	size= candidate_query_points_num* sizeof(int)* K_NN;
	err = cudaMemcpy(idx_k_mins_global_tmp, d_idx_k_mins_global_tmp, size, cudaMemcpyDeviceToHost);
	err = cudaGetLastError();
	if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
        //exit(EXIT_FAILURE);
    }
	
	free_cuda_mem();
	return 0;
}

/**
* Host main routine
*/
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
	int					DIM)
{
	clock_t start, finish, start1, finish1;
	float duration;
	bool cuda_init = init_CUDA_device();
	if (cuda_init){
		printf("\nsucced for initializing CUDA");
	}
	cudaError_t err = cudaSuccess;

	//Launch the Vector Add CUDA Kernel
	printf("\nCUDA Malloc Memory.....\n ");

	start = clock();
	size_t size = candidate_query_points_num * sizeof(int);
	if (d_candidate_query_points_indexes == NULL){
		err = cudaMalloc((void **)&d_candidate_query_points_indexes, size);
		err = cudaMemcpy(d_candidate_query_points_indexes, candidate_query_points_indexes, size, cudaMemcpyHostToDevice);
	}

	size = candidate_query_points_num * sizeof(FLOAT_TYPE)*DIM;
	if (d_candidate_query_points_set == NULL){
		err = cudaMalloc((void **)&d_candidate_query_points_set, size);
		err = cudaMemcpy(d_candidate_query_points_set, candidate_query_points_set, size, cudaMemcpyHostToDevice);
	}

	size = sorted_data_len * sizeof(FLOAT_TYPE)*DIM;
	if (d_all_sorted_data_set == NULL){
		err = cudaMalloc((void **)&d_all_sorted_data_set, size);
		err = cudaMemcpy(d_all_sorted_data_set, all_sorted_data_set, size, cudaMemcpyHostToDevice);
	}


	size = sorted_data_len * sizeof(int);
	if (d_sorted_data_set_indexes == NULL){
		err = cudaMalloc((void **)&d_sorted_data_set_indexes, size);
		err = cudaMemcpy(d_sorted_data_set_indexes, sorted_data_set_indexes, size, cudaMemcpyHostToDevice);
	}

	size = tree_nodes_num*sizeof(CONVEX_TREE);
	if (d_tree_struct == NULL){
		err = cudaMalloc((void **)&d_tree_struct, size);
		err = cudaMemcpy(d_tree_struct, tree_struct, size, cudaMemcpyHostToDevice);
	}

	size = all_leaf_nodes_constraint_num* sizeof(FLOAT_TYPE)*DIM;
	if (d_all_leaf_nodes_ALPHA_set == NULL){
		err = cudaMalloc((void **)&d_all_leaf_nodes_ALPHA_set, size);
		err = cudaMemcpy(d_all_leaf_nodes_ALPHA_set, all_leaf_nodes_ALPHA_set, size, cudaMemcpyHostToDevice);
	}

	size = all_leaf_nodes_constraint_num* sizeof(FLOAT_TYPE);
	if (d_all_leaf_nodes_BETA_set == NULL){
		err = cudaMalloc((void **)&d_all_leaf_nodes_BETA_set, size);
		err = cudaMemcpy(d_all_leaf_nodes_BETA_set, all_leaf_nodes_BETA_set, size, cudaMemcpyHostToDevice);
	}

	size = leaf_node_num*sizeof(int);
	if (d_all_constrains_num_of_each_leaf_nodes == NULL){
		err = cudaMalloc((void **)&d_all_constrains_num_of_each_leaf_nodes, size);
		err = cudaMemcpy(d_all_constrains_num_of_each_leaf_nodes, all_constrains_num_of_each_leaf_nodes, size, cudaMemcpyHostToDevice);
	}

	if (d_all_leaf_nodes_offsets_in_all_ALPHA == NULL){
		err = cudaMalloc((void **)&d_all_leaf_nodes_offsets_in_all_ALPHA, size);
		err = cudaMemcpy(d_all_leaf_nodes_offsets_in_all_ALPHA, all_leaf_nodes_offsets_in_all_ALPHA, size, cudaMemcpyHostToDevice);
	}

	if (d_all_leaf_nodes_ancestor_nodes_ids == NULL){
		err = cudaMalloc((void **)&d_all_leaf_nodes_ancestor_nodes_ids, size);
		err = cudaMemcpy(d_all_leaf_nodes_ancestor_nodes_ids, all_leaf_nodes_ancestor_nodes_ids, size, cudaMemcpyHostToDevice);
	}

	if (d_leaf_nodes_start_pos_in_sorted_data_set == NULL){
		err = cudaMalloc((void **)&d_leaf_nodes_start_pos_in_sorted_data_set, size);
		err = cudaMemcpy(d_leaf_nodes_start_pos_in_sorted_data_set, leaf_nodes_start_pos_in_sorted_data_set, size, cudaMemcpyHostToDevice);
	}

	if (d_pts_num_in_sorted_leaf_nodes == NULL){
		err = cudaMalloc((void **)&d_pts_num_in_sorted_leaf_nodes, size);
		err = cudaMemcpy(d_pts_num_in_sorted_leaf_nodes, pts_num_in_sorted_leaf_nodes, size, cudaMemcpyHostToDevice);
	}


	size = candidate_query_points_num* sizeof(FLOAT_TYPE)* K_NN;
	if (d_dist_k_mins_global_tmp == NULL){
		err = cudaMalloc((void **)&d_dist_k_mins_global_tmp, size);
		err = cudaMemcpy(d_dist_k_mins_global_tmp, dist_k_mins_global_tmp, size, cudaMemcpyHostToDevice);
	}

	size = candidate_query_points_num* sizeof(int)* K_NN;
	if (d_idx_k_mins_global_tmp == NULL){
		err = cudaMalloc((void **)&d_idx_k_mins_global_tmp, size);
		err = cudaMemcpy(d_idx_k_mins_global_tmp, idx_k_mins_global_tmp, size, cudaMemcpyHostToDevice);
	}
	size = sorted_data_len * sizeof(int);
	int* d_remain_index = NULL;
	err = cudaMalloc((void **)&d_remain_index, size);
	err = cudaMemcpy(d_remain_index, remain_index, size, cudaMemcpyHostToDevice);

	size = candidate_query_points_num*sizeof(long);
	err = cudaMalloc((void **)&d_dist_computation_times_arr, size);

	err = cudaMalloc((void **)&d_quadprog_times_arr, size);

	err = cudaMalloc((void **)&d_dist_computation_times_in_quadprog, size);
	int task_per_num = 100000 / 256;

	finish = clock();
	duration = (float)(finish - start) / CLOCKS_PER_SEC;
	//printf( "\n CUDA Malloc Memory Time %f\n", duration);

	//printf ("\n calling do_finding_KNN_by_leaf_order_cuda.....\n");
	int block_num = candidate_query_points_num / 1024 + 1;
	dim3  blocks(block_num, 1), threads(1024, 1);
	//int blocks_per_time = 6 * 16 * 128 * 8;

	start1 = clock();
	new_do_finding_KNN_by_leaf_order_cuda << < blocks, threads >> >(
		candidate_query_points_num,
		d_candidate_query_points_indexes,
		d_candidate_query_points_set,
		d_all_sorted_data_set,
		d_sorted_data_set_indexes,
		d_tree_struct,
		d_all_leaf_nodes_ALPHA_set,
		d_all_leaf_nodes_BETA_set,
		d_all_constrains_num_of_each_leaf_nodes,
		d_all_leaf_nodes_offsets_in_all_ALPHA,
		leaf_node_num,
		d_all_leaf_nodes_ancestor_nodes_ids,
		d_leaf_nodes_start_pos_in_sorted_data_set,
		d_pts_num_in_sorted_leaf_nodes,
		d_dist_k_mins_global_tmp,
		d_idx_k_mins_global_tmp,
		d_remain_index,
		K_NN,
		d_dist_computation_times_arr,
		d_quadprog_times_arr,
		d_dist_computation_times_in_quadprog,
		NODES_NUM,
		DIM,
		1);

	err = cudaGetLastError();
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
		//exit(EXIT_FAILURE);
	}
	finish1 = clock();
	duration = (float)(finish1 - start1) / CLOCKS_PER_SEC;
	//printf( "\n performing do_finding_KNN_by_leaf_order_cuda time %f milsec %f s\n",(float)(finish1-start1), duration);

	//printf( "----print device matrix-------");
	//print_float_Data <<<4, 256>>>(d_dist_k_mins_global_tmp,d_idx_k_mins_global_tmp,0);


	// Copy the device result vector in device memory to the host result vector
	// in host memory.

	//printf("\n copy data from GPU....\n");
	size = candidate_query_points_num* sizeof(FLOAT_TYPE)* K_NN;
	err = cudaMemcpy(dist_k_mins_global_tmp, d_dist_k_mins_global_tmp, size, cudaMemcpyDeviceToHost);

	size = candidate_query_points_num* sizeof(int)* K_NN;
	err = cudaMemcpy(idx_k_mins_global_tmp, d_idx_k_mins_global_tmp, size, cudaMemcpyDeviceToHost);
	err = cudaGetLastError();
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
		//exit(EXIT_FAILURE);
	}

	free_cuda_mem();
	return 0;
}


__global__ void kernel_do_find_approximate_nodes(
												int          candidate_query_points_num,
										 FLOAT_TYPE         *candidate_query_points_set,
												int          tree_nodes_num,
										CONVEX_TREE         *tree_struct,
										 FLOAT_TYPE         *nodes_centers,     
												int         *appr_leaf_node_indexes,
												int          DIM
												){
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid >= candidate_query_points_num) return;
	
	FLOAT_TYPE *q = candidate_query_points_set + tid * DIM;
	int cur_node_index=0;
	while (tree_struct[cur_node_index].isLeaf == false){		
		FLOAT_TYPE left_dist_squre = float_dist_squre_cuda(q, nodes_centers+tree_struct[cur_node_index].left_node*DIM,DIM);
		FLOAT_TYPE right_dist_squre = float_dist_squre_cuda(q, nodes_centers+tree_struct[cur_node_index].right_node*DIM ,DIM);
		//count the distance computation times.
		if (left_dist_squre >= right_dist_squre){
			cur_node_index = tree_struct[cur_node_index].right_node;
		}
		else{
			cur_node_index = tree_struct[cur_node_index].left_node;
		}
	}
	appr_leaf_node_indexes[tid] = tree_struct[cur_node_index].leaf_index;
}


extern "C" void call_do_find_approximate_nodes(
												int          candidate_query_points_num,
										 FLOAT_TYPE         *candidate_query_points_set,
												int          tree_nodes_num,
										CONVEX_TREE         *tree_struct,
										 FLOAT_TYPE         *nodes_centers,     
												int         *candidate_query_points_appr_leaf_node_indexes,
												int          DIM
												){
	
	size_t size = candidate_query_points_num * sizeof(int);
	cudaError_t err = cudaSuccess;
	size = candidate_query_points_num * sizeof(FLOAT_TYPE)*DIM;
	err = cudaMalloc((void **)&d_candidate_query_points_set, size);
	err = cudaMemcpy(d_candidate_query_points_set, candidate_query_points_set, size, cudaMemcpyHostToDevice);

	size = tree_nodes_num * sizeof(CONVEX_TREE);
	err = cudaMalloc((void **)&d_tree_struct, size);
	err = cudaMemcpy(d_tree_struct, tree_struct, size, cudaMemcpyHostToDevice);

	size = candidate_query_points_num * sizeof(int);
	err = cudaMalloc((void **)&d_candidate_query_points_appr_leaf_node_indexes, size);

	size= tree_nodes_num*sizeof(FLOAT_TYPE)*DIM;
	err = cudaMalloc((void **)&d_nodes_centers, size);
	err = cudaMemcpy(d_nodes_centers, nodes_centers, size, cudaMemcpyHostToDevice);
	
	int block_num =candidate_query_points_num/1024 +1;
	dim3  blocks(block_num,1), threads(1024,1); 
	kernel_do_find_approximate_nodes<<<blocks, threads>>>(  candidate_query_points_num,
															d_candidate_query_points_set,
															tree_nodes_num,
															d_tree_struct,
															d_nodes_centers,     
															d_candidate_query_points_appr_leaf_node_indexes,
															DIM);
	err = cudaGetLastError();
	if (err != cudaSuccess){
		printf("Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
	}

	size = candidate_query_points_num * sizeof(int);
	err = cudaMemcpy(candidate_query_points_appr_leaf_node_indexes, d_candidate_query_points_appr_leaf_node_indexes, size, cudaMemcpyDeviceToHost);
	
	err = cudaGetLastError();
	if (err != cudaSuccess){
		printf("Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
	}

}

/**
* Host main routine
*/
/*
extern "C" int call_cuda_kernel_brute_force_and_update(
				int          candidate_query_points_num,
				int         *candidate_query_points_indexes,
		 FLOAT_TYPE         *candidate_query_points_set,
				int          data_set_size,
		 FLOAT_TYPE         *data_set,
		 FLOAT_TYPE         *KNN_index_with_dist,
				int          K_NN,
				int          DIM)
{
	clock_t start, finish, start1, finish1;
	bool cuda_init = init_CUDA_device();
	if (cuda_init){
		printf("succed for initializing CUDA\n");
	}
	cudaError_t err = cudaSuccess;

	//Launch the Vector Add CUDA Kernel
	printf("CUDA Malloc Memory.....\n");

	start = clock();
	size_t size = candidate_query_points_num * sizeof(int);
	int *d_candidate_query_points_indexes = NULL;
	err = cudaMalloc((void **)&d_candidate_query_points_indexes, size);
	err = cudaMemcpy(d_candidate_query_points_indexes, candidate_query_points_indexes, size, cudaMemcpyHostToDevice);

	size = candidate_query_points_num * sizeof(FLOAT_TYPE)*DIM;
	FLOAT_TYPE *d_candidate_query_points_set = NULL;
	err = cudaMalloc((void **)&d_candidate_query_points_set, size);
	err = cudaMemcpy(d_candidate_query_points_set, candidate_query_points_set, size, cudaMemcpyHostToDevice);

	size = data_set_size * sizeof(FLOAT_TYPE)*DIM;
	FLOAT_TYPE *d_data_set = NULL;
	err = cudaMalloc((void **)&d_data_set, size);
	err = cudaMemcpy(d_data_set, data_set, size, cudaMemcpyHostToDevice);
	
	size = candidate_query_points_num* sizeof(FLOAT_TYPE)* K_NN * 2;
	FLOAT_TYPE *d_KNN_index_with_dist = NULL;
	err = cudaMalloc((void **)&d_KNN_index_with_dist, size);

	finish = clock();
	double duration = (double)(finish - start) / CLOCKS_PER_SEC;
	printf("CUDA Malloc Memory Time %fs\n", duration);


	printf("calling do_finding_KNN_by_leaf_order_cuda.....\n");
	dim3 grids(2, 1), blocks(6, 1), threads(1024, 1);
	start1 = clock();

	do_brute_force_KNN_cuda << < blocks, threads >> >
		(d_data_set, data_set_size, d_candidate_query_points_set, candidate_query_points_num,
		d_KNN_index_with_dist, K_NN, DIM);

	err = cudaGetLastError();
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
		//exit(EXIT_FAILURE);
	}

	//printf( "----print device matrix-------");
	//print_float_Data <<<4, 256>>>(d_dist_k_mins_global_tmp,d_idx_k_mins_global_tmp,0);


	// Copy the device result vector in device memory to the host result vector
	// in host memory.


	//printf("\n copy data from GPU....\n");
	size = candidate_query_points_num* sizeof(FLOAT_TYPE)* K_NN * 2;
	err = cudaMemcpy(KNN_index_with_dist, d_KNN_index_with_dist, size, cudaMemcpyDeviceToHost);
	//size= candidate_query_points_num* sizeof(FLOAT_TYPE)* K_NN;
	//err = cudaMemcpy(dist_k_mins_global_tmp, d_dist_k_mins_global_tmp, size, cudaMemcpyDeviceToHost);

	//size= candidate_query_points_num* sizeof(int)* K_NN;
	//err = cudaMemcpy(idx_k_mins_global_tmp, d_idx_k_mins_global_tmp, size, cudaMemcpyDeviceToHost);
	
	err = cudaGetLastError();
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
		//exit(EXIT_FAILURE);
	}
	finish1 = clock();
	duration = (double)(finish1 - start1) / CLOCKS_PER_SEC;
	printf("performing do_finding_KNN_by_leaf_order_cuda time %fs\n", duration);

	return 0;
}
*/

/**
 * Host main routine
 */
extern "C" int call_cuda_kernel_brute_force(
				int          candidate_query_points_num,
			    int         *candidate_query_points_indexes,
		 FLOAT_TYPE         *candidate_query_points_set,
				int          data_set_size,
		 FLOAT_TYPE         *data_set,
	     FLOAT_TYPE         *dist_k_mins_global_tmp,
		        int         *idx_k_mins_global_tmp,
				int          K_NN,
			   	int          DIM)
{
	clock_t start, finish,start1, finish1;
	bool cuda_init =init_CUDA_device();
	if (cuda_init){
	    printf("succed for initializing CUDA\n");
	}
	cudaError_t err = cudaSuccess;

	//Launch the Vector Add CUDA Kernel
    printf("CUDA Malloc Memory.....\n");

	start=clock();
	size_t size = candidate_query_points_num * sizeof(int);
	int *d_candidate_query_points_indexes = NULL;	
    err = cudaMalloc((void **)&d_candidate_query_points_indexes, size);
	err = cudaMemcpy(d_candidate_query_points_indexes, candidate_query_points_indexes, size, cudaMemcpyHostToDevice);

	for (int i = 0; i < 100; i++){
		for (int j = 0; j < DIM; j++){
			printf("%.1f  ", candidate_query_points_set[i*DIM + j]);
		}
		printf("\n");
	}

	size = candidate_query_points_num * sizeof(FLOAT_TYPE)*DIM;
	FLOAT_TYPE *d_candidate_query_points_set=NULL;
	err = cudaMalloc((void **)&d_candidate_query_points_set, size);
	err = cudaMemcpy(d_candidate_query_points_set, candidate_query_points_set, size, cudaMemcpyHostToDevice);

	size = data_set_size * sizeof(FLOAT_TYPE)*DIM;
	FLOAT_TYPE *d_data_set=NULL;
	err = cudaMalloc((void **)&d_data_set, size);
	err = cudaMemcpy(d_data_set, data_set, size, cudaMemcpyHostToDevice);

	size= candidate_query_points_num* sizeof(FLOAT_TYPE)* K_NN;
	FLOAT_TYPE *d_dist_k_mins_global_dist = NULL;
	err = cudaMalloc((void **)&d_dist_k_mins_global_dist, size);

	size= candidate_query_points_num* sizeof(int)* K_NN;
	int  *d_idx_k_mins_global_tmp=NULL;
	err = cudaMalloc((void **)&d_idx_k_mins_global_tmp, size);

	
	finish=clock();
	double duration = (double)(finish-start)/ CLOCKS_PER_SEC;
	printf( "CUDA Malloc Memory Time %fs\n", duration);

	
	printf ("calling do_finding_KNN_by_leaf_order_cuda.....\n");
	int block_num = candidate_query_points_num / 1024 + 1;
	dim3  blocks(block_num, 1), threads(1024, 1);
	start1=clock();
	
	
	do_brute_force_KNN_cuda<<< blocks, threads>>>
	                       (d_data_set,  data_set_size,	d_candidate_query_points_set, candidate_query_points_num,
						   d_dist_k_mins_global_dist, d_idx_k_mins_global_tmp, K_NN, DIM);

    err = cudaGetLastError();
	if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
        //exit(EXIT_FAILURE);
    }
	printf("ending......\n");
	
	size= candidate_query_points_num* sizeof(FLOAT_TYPE)* K_NN;
	err = cudaMemcpy(dist_k_mins_global_tmp, d_dist_k_mins_global_tmp, size, cudaMemcpyDeviceToHost);

	size= candidate_query_points_num* sizeof(int)* K_NN;
	err = cudaMemcpy(idx_k_mins_global_tmp, d_idx_k_mins_global_tmp, size, cudaMemcpyDeviceToHost);

	err = cudaGetLastError();
	if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
        //exit(EXIT_FAILURE);
    }
	finish1=clock();
	duration = (double)(finish1-start1)/ CLOCKS_PER_SEC;
	printf( "performing do_finding_KNN_by_leaf_order_cuda time %fs\n", duration);

	return 0;
}

/**
* Host main routine
*/
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
	int                DIM)
{
	clock_t start, finish, start1, finish1;
	bool cuda_init = init_CUDA_device();
	/*
	if (cuda_init){
		printf("succed for initializing CUDA\n");
	}
	*/
	cudaError_t err = cudaSuccess;

	//Launch the Vector Add CUDA Kernel
	//printf("CUDA Malloc Memory.....\n");

	start = clock();
	size_t size = candidate_query_points_num * sizeof(int);
	int *d_candidate_query_points_indexes = NULL;
	err = cudaMalloc((void **)&d_candidate_query_points_indexes, size);
	err = cudaMemcpy(d_candidate_query_points_indexes, candidate_query_points_indexes, size, cudaMemcpyHostToDevice);

	size = candidate_query_points_num * sizeof(FLOAT_TYPE)*DIM;
	FLOAT_TYPE *d_candidate_query_points_set = NULL;
	err = cudaMalloc((void **)&d_candidate_query_points_set, size);
	err = cudaMemcpy(d_candidate_query_points_set, candidate_query_points_set, size, cudaMemcpyHostToDevice);

	size = data_set_size * sizeof(FLOAT_TYPE)*DIM;
	FLOAT_TYPE *d_data_set = NULL;
	err = cudaMalloc((void **)&d_data_set, size);
	err = cudaMemcpy(d_data_set, data_set, size, cudaMemcpyHostToDevice);

	size = data_set_size * sizeof(int);
	int *d_data_indexes = NULL;
	err = cudaMalloc((void **)&d_data_indexes, size);
	err = cudaMemcpy(d_data_indexes, data_indexes, size, cudaMemcpyHostToDevice);

	size = candidate_query_points_num* sizeof(FLOAT_TYPE)* K_NN;
	FLOAT_TYPE *d_dist_k_mins_global_dist = NULL;
	err = cudaMalloc((void **)&d_dist_k_mins_global_dist, size);
	err = cudaMemcpy(d_dist_k_mins_global_dist, dist_k_mins_global_tmp, size, cudaMemcpyHostToDevice);

	size = candidate_query_points_num* sizeof(int)* K_NN;
	int  *d_idx_k_mins_global_tmp = NULL;
	err = cudaMalloc((void **)&d_idx_k_mins_global_tmp, size);
	err = cudaMemcpy(d_idx_k_mins_global_tmp, idx_k_mins_global_tmp, size, cudaMemcpyHostToDevice);


	finish = clock();
	double duration = (double)(finish - start) / CLOCKS_PER_SEC;
	//printf("CUDA Malloc Memory Time %fs\n", duration);


	//printf("calling do_finding_KNN_by_leaf_order_cuda.....\n");
	int block_num = candidate_query_points_num / 1024 + 1;
	dim3  blocks(block_num, 1), threads(1024, 1);
	start1 = clock();

	do_brute_force_KNN_for_outliers_cuda << < blocks, threads >> >
		(d_data_set, data_set_size, d_data_indexes, d_candidate_query_points_set, candidate_query_points_num,
		d_dist_k_mins_global_dist, d_idx_k_mins_global_tmp, K_NN, DIM);

	err = cudaGetLastError();
	if (err != cudaSuccess)
	{
		//fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
		//exit(EXIT_FAILURE);
	}

	//printf( "----print device matrix-------");
	//print_float_Data <<<4, 256>>>(d_dist_k_mins_global_tmp,d_idx_k_mins_global_tmp,0);


	// Copy the device result vector in device memory to the host result vector
	// in host memory.


	//printf("\n copy data from GPU....\n");

	size = candidate_query_points_num* sizeof(FLOAT_TYPE)* K_NN;
	err = cudaMemcpy(dist_k_mins_global_tmp, d_dist_k_mins_global_tmp, size, cudaMemcpyDeviceToHost);

	size = candidate_query_points_num* sizeof(int)* K_NN;
	err = cudaMemcpy(idx_k_mins_global_tmp, d_idx_k_mins_global_tmp, size, cudaMemcpyDeviceToHost);

	err = cudaGetLastError();
	if (err != cudaSuccess)
	{
		//fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
		//exit(EXIT_FAILURE);
	}
	finish1 = clock();
	duration = (double)(finish1 - start1) / CLOCKS_PER_SEC;
	//printf("performing do_finding_KNN_by_leaf_order_cuda time %fs\n", duration);

	return 0;
}
