#include <iostream>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <vector>
#include "cover_tree.h"
using namespace std;

//Compute the k nearest neighbors
struct Point_K{
	float p, q;
	float d;//d is distance between p and q
};

struct node_p{
	int index;
	float value;
};

float* generate_query_data(float* source_data, int batch, int batch_num, int dim){
	float* query_data = new float[batch_num*dim];
	for (int i = batch*batch_num; i < (batch + 1)*batch_num; i++){
		for (int j = 0; j < dim; j++){
			query_data[(i - batch*batch_num)*dim + j] = source_data[i*dim + j];
		}
	}
	//cout << "Generate query data:" << batch*batch_num << "--" << (batch + 1)*batch_num - 1 << endl;
	return query_data;
}

bool cmp(struct node_p a, struct node_p b){
	if (a.value < b.value){
		return true;
	}
	return false;
}

void quickSort(struct Point_K s[], int l, int r){
	if (l< r)
	{
		int i = l, j = r;
		struct Point_K x = s[l];
		while (i < j)
		{
			while (i < j && s[j].d >= x.d) // 从右向左找第一个小于x的数
				j--;
			if (i < j)
				s[i++] = s[j];
			while (i < j && s[i].d < x.d) // 从左向右找第一个大于等于x的数
				i++;
			if (i < j)
				s[j--] = s[i];
		}
		s[i] = x;
		quickSort(s, l, i - 1); // 递归调用
		quickSort(s, i + 1, r);
	}
}

void Point_Sort(float* data, int K){
	Point_K *p = (Point_K *)malloc(K*sizeof(Point_K));
	for (int i = 0; i < K; i++){
		p[i].p = data[i * 3];
		p[i].q = data[i * 3 + 1];
		p[i].d = data[i * 3 + 2];
	}
	quickSort(p, 0, K - 1);
	for (int i = 0; i < K; i++){
		data[i * 3] = p[i].p;
		data[i * 3 + 1] = p[i].q;
		data[i * 3 + 2] = p[i].d;
	}
}

void ComputeDistance(v_array<v_array<float> >* result, int dim, int batch, int batch_num, int K, float* distance_re, float* raw_data){
	int k = 0, p_d = dim - 1, p1_index, p2_index;//p_d is the dim of point except index
	float d;//d is the distance of p1 between p2
	float *p1, *p2;
	for (int i = 0; i < batch; i++){
		for (int j = 0; j < result[i].index; j++){
			p1_index = (int)result[i][j][0];
			p1 = raw_data + dim*p1_index;
			for (int k = 1; k < result[i][j].index&&k <= K; k++){
				p2_index = (int)result[i][j][k];
				p2 = raw_data + dim*p2_index;
				d = distance(p1, p2, p_d);
				distance_re[3 * K*p1_index + 3 * (k - 1)] = p1_index;
				distance_re[3 * K*p1_index + 3 * (k - 1) + 1] = p2_index;
				distance_re[3 * K*p1_index + 3 * (k - 1) + 2] = d;
			}
		}
	}
}

void PretreatDistance(float* dis_matrix, int data_size, int K){
	float* temp = new float[3 * K];
	for (int i = 0; i < data_size; i++){
		memcpy(temp, dis_matrix + 3 * K*i, (3 * K)*sizeof(float));
		Point_Sort(temp, K);
		memcpy(dis_matrix + 3 * K*i, temp, (3 * K)*sizeof(float));
	}
}

float* ComputeDensity(float* dis_matrix, int data_size, int K){
	float* density = new float[data_size];
	for (int i = 0; i < data_size; i++){
		density[i] = 1.0 / dis_matrix[3 * (i + 1)*K - 1];
	}
	return density;
}

int* FindCluster(float* density, float* delta, int cl, int data_size){
	float* r = (float*)malloc(data_size*sizeof(float));
	for (int i = 0; i < data_size; i++)
		r[i] = density[i] * delta[i];
	int* cluster = (int*)malloc(cl*sizeof(int));
	for (int i = 0; i < cl; i++){
		cluster[i] = 0;
		for (int j = 1; j < data_size; j++){
			int flag = -1, cl = cluster[i];
			if (r[j] > r[cl]){
				for (int k = 0; k < i; k++){
					if (j == cluster[k])  flag = 0;
				}
				if (flag == -1)
					cluster[i] = j;
			}
		}
	}
	return cluster;
}

float Find_threshold(float* a, int d_size, float percent){
	float* b = (float*)malloc(d_size*sizeof(float));
	memcpy(b, a, d_size*sizeof(float));
	sort(b, b + d_size);
	int pos = floor(d_size*percent);
	float result = b[pos];
	free(b);
	return result;
}

void Find_tem_core(float* dis_matrix, float* density, int* tem_Core, float* delta, int data_size, int K){
	for (int i = 0; i < data_size; i++){
		int p1 = dis_matrix[i*K * 3];
		for (int j = 1; j < K; j++){
			int p2 = dis_matrix[i*K * 3 + j * 3 + 1];
			if (density[p2] > density[p1] && p2 < data_size){
				tem_Core[i] = p2;
				delta[i] = dis_matrix[i*K * 3 + j * 3 + 2];
				break;
			}
		}
	}
}

long PreProcess_local_density_peak(float* density, float* raw_data, float* dis_matrix, float* delta, int* tem_Core,
	                               int local_peak_threshold, int data_size, int K, int dim, node node_data){
	int p_d = dim - 1;//p_d is the dim of point except index
	int local_peak_num = 0;
	long c_d_n = 0;//c_d_n is the number of computing distance
	float *p1, *p2, d;
	//count the number of local density peak
	for (int i = 0; i < data_size; i++){
		if (tem_Core[i] == -1)
			local_peak_num++;
	}
	if (local_peak_num > local_peak_threshold){
		int fre = 1;//fre is the frequency of loop
		while (local_peak_num > local_peak_threshold){
			std::cout << "local_peak_num:" << local_peak_num << endl;
			int* peak_index = (int *)malloc(local_peak_num*sizeof(int));
			int num = 0;
			//array peak_index stores the index of local density peaks
			for (int i = 0; i < data_size; i++){
				if (tem_Core[i] == -1)
					peak_index[num++] = i;
			}
			//array peak_data stores the data of local density peaks
			float* peak_data = (float *)malloc(local_peak_num*dim*sizeof(float));
			for (int i = 0; i < local_peak_num; i++)
				memcpy(peak_data + i*dim, raw_data + peak_index[i] * dim, dim*sizeof(float));
			//building tree for local density peaks
			v_array<point> peak_queries = parse_points(peak_data, local_peak_num, dim);
			node node_query = batch_create(peak_queries);

			v_array<v_array<float> > result;
			k_nearest_neighbor(node_data, node_query, result, K*(fre * 2), dim);

			int p1_index, p2_index, K_dis_threshold, flag;
			for (int i = 0; i < result.index; i++){
				p1_index = (int)result[i][0];
				p1 = raw_data + p1_index*dim;
				K_dis_threshold = dis_matrix[(p1_index + 1) * 3 * K - 1];
				flag = 0;
				for (int j = 1; j < result[i].index&&j <= (fre + 1)*K; j++){
					p2_index = (int)result[i][j];
					p2 = raw_data + p2_index*dim;
					d = distance(p1, p2, p_d);
					c_d_n++;
					if (d > K_dis_threshold&&density[p2_index] > density[p1_index]){
						if (flag == 0){
							tem_Core[p1_index] = p2_index;
							delta[p1_index] = d;
							flag = 1;
						}
						else{
							if (d < delta[p1_index]){
								tem_Core[p1_index] = p2_index;
								delta[p1_index] = d;
							}
						}
					}
				}
				if (tem_Core[p1_index] != -1)
					local_peak_num--;
			}
			fre++;
			free(peak_data);
			free(peak_index);
		}
	}
	return c_d_n;
}

int Find_local_density_peak(float* density, int* tem_Core, float* delta, float* raw_data, int local_peak_threshold, long long* C_D_N,
	                        float* dis_matrix, int data_size, int dim, int K, node node_data, node_p* node_p_ptr){
	node_p * p = node_p_ptr;
	for (int i = 0; i < data_size; i++){
		p[i].index = i;
		p[i].value = density[i];
	}
	sort(p, p + data_size, cmp);

	long c_d_n = PreProcess_local_density_peak(density, raw_data, dis_matrix, delta, tem_Core, local_peak_threshold, data_size, K, dim, node_data);

	int p_d = dim - 1;//p_d is the dim of point except index
	int local_density_peak_num = 0;
	long long compute_distance_num = c_d_n;
	float *p1, *p2, den;
	int* flag = (int *)malloc(data_size*sizeof(int));
	for (int i = 0; i < data_size; i++){
		if (tem_Core[i] == -1){
			memset(flag, 0, data_size*sizeof(int));
			local_density_peak_num++;
			int den_order = 0;
			//find the order of point i by density
			for (int j = 0; j < data_size; j++){
				if (i == p[j].index){
					den_order = j;
				}
			}
			//if point den_order of density is the largest than other points
			if (den_order == data_size - 1){
				float max_dis = 0;
				for (int j = 0; j < data_size; j++){
					p1 = raw_data + i*dim;
					p2 = raw_data + j*dim;
					float d = distance(p1, p2, p_d);
					if (max_dis < d)
						max_dis = d;
				}
				delta[i] = max_dis;
				continue;
			}
			p1 = raw_data + (p[den_order].index)*dim;
			float local_nearest_distance = 1000000;
			int local_nearest_index = -1;
			for (int k = den_order + 1; k < data_size; k++){
				int local_index = p[k].index;
				//if local point is not be filtered
				if (flag[local_index] == 0 && density[i] <= density[local_index]){
					p2 = raw_data + local_index*dim;
					float d = distance(p1, p2, p_d);
					compute_distance_num++;
					//if distance between local point and local density peak is smaller than local nearest distance
					if (d < local_nearest_distance){
						local_nearest_distance = d;
						local_nearest_index = local_index;
						continue;
					}
					//filter some points by triangle inequality
					for (int j = K; j > 0; j--){
						float d_p_j = dis_matrix[local_index * 3 * K + 3 * j - 1];
						if (d > local_nearest_distance + d_p_j){
							for (int s = 1; s <= j; s++){
								int index = dis_matrix[local_index * 3 * K + 3 * s - 2];
								flag[index] = 1;
							}
							break;
						}
					}
				}
			}
			tem_Core[i] = local_nearest_index;
			delta[i] = local_nearest_distance;
		}
	}
	*C_D_N = compute_distance_num;

	free(flag);
	free(p);
	return local_density_peak_num;
}

void Label_cluster(int* Point_cl, int* tem_Core, int* cluster, int cl, int data_size){
	int c = 0;
	for (int i = 0; i < data_size; i++){
		int flag = -1;//Not find Core in cluster
		int p = i;
		while (flag == -1){
			for (int j = 0; j < cl; j++){
				if (tem_Core[p] == cluster[j] || p == cluster[j]){
					flag = 0;//Find Core in cluster
					c = j + 1;
				}
			}
			if (flag == 0)
				Point_cl[i] = c;
			else
				p = tem_Core[p];
		}
	}
}

int* Find_Exc(float* density, float* delta, float density_threshold, float delta_threshold, int data_size, int* exc_n){
	stack<int> Exc;
	for (int i = 0; i < data_size; i++){
		if (density[i] < density_threshold && delta[i] > delta_threshold)
			Exc.push(i);
	}
	*exc_n = Exc.size();
	int* Exc_index = (int*)malloc(*exc_n*sizeof(int));
	for (int i = 0; i < *exc_n; i++){
		Exc_index[i] = Exc.top();
		Exc.pop();
	}
	return Exc_index;
}

int* Find_Exc_plus(int* tem_Core, int* Exc, int data_size, int exc_num, int* Exc_plus_n){
	stack<int> Exc_plus;
	for (int i = 0; i < data_size; i++){
		int flag = -1, p = i;
		while (flag == -1 && p != -1){
			for (int j = 0; j < exc_num; j++){
				if (tem_Core[p] == Exc[j] || p == Exc[j]){
					flag = 0;
					break;
				}
			}
			if (flag == 0)
				Exc_plus.push(i);
			else
				p = tem_Core[p];
		}
	}
	*Exc_plus_n = Exc_plus.size();
	int* Exc_plus_index = (int *)malloc(*Exc_plus_n*sizeof(int));
	for (int i = 0; i < *Exc_plus_n; i++){
		Exc_plus_index[i] = Exc_plus.top();
		Exc_plus.pop();
	}
	return Exc_plus_index;
}

void Fast_Density_Peak(int K, float* raw_data, int data_size, int batch_num, int dim, int* Exc_data_index,
	                   int* exc_n, int local_peak_threshold, float* dis_matrix, node_p *node_p_ptr){
	//std::cout << "\n-------------------------------------------------------------------" << endl;
	//std::cout << "Starting FastDPeak........." << endl;
	dim++;
	v_array<point> data_set = parse_points(raw_data, data_size, dim);

	//printf("building tree for source data....\n");
	node node_data = batch_create(data_set);

	int batch = data_size / batch_num;
	//std::cout << "batch:" << batch << endl;

	v_array<v_array<float> >* res = new v_array<v_array<float> >[batch];
	for (int j = 0; j<batch; j++){
		v_array<v_array<float> >* tmp_batch = new v_array<v_array<float> >(batch_num);
		res[j] = *tmp_batch;
		for (int i = 0; i<batch_num; i++){
			v_array<float> tmp_v(K);
			push(res[j], tmp_v);
		}
		res[j].index = 0;
	}

	for (int i = 0; i < batch; i++){
		//std::cout << "generate queries...\n";
		float* query_data = generate_query_data(raw_data, i, batch_num, dim);
		v_array<point> queries = parse_points(query_data, batch_num, dim);

		node node_query = batch_create(queries);

		k_nearest_neighbor_new(node_data, node_query, res[i], K, dim);
		queries.free_resource();
	}
	data_set.free_resource();

	ComputeDistance(res, dim, batch, batch_num, K, dis_matrix, raw_data);

	for (int i = 0; i < res->length; i++){
		res->elements[i].free_resource();
	}
	res->free_resource();

	int d_size = batch * batch_num;
	PretreatDistance(dis_matrix, d_size, K);

	float* density = ComputeDensity(dis_matrix, d_size, K);

	int* tem_core = (int *)malloc(d_size*sizeof(int));
	float* delta = (float *)malloc(d_size*sizeof(float));
	memset(tem_core, -1, d_size*sizeof(int));
	memset(delta, -1, d_size*sizeof(float));

	Find_tem_core(dis_matrix, density, tem_core, delta, d_size, K);

	long long C_D_N = 0;
	int LDP_N = 0;

	LDP_N = Find_local_density_peak(density, tem_core, delta, raw_data, local_peak_threshold, &C_D_N, dis_matrix, d_size, dim, K, node_data, node_p_ptr);
	//Find_local_density_peak(density,tem_core,delta,raw_data,local_peak_threshold,dis_matrix,d_size,dim,K,buffer_l,node_data, node_p_ptr);

	float den_threshold = Find_threshold(density, d_size, 0.01);
	//cout << "density_threshold:" << den_threshold << endl;
	float del_threshold = Find_threshold(delta, d_size, 0.95);
	//cout << "delta_threshold:" << del_threshold << endl;

	/*
	for (int i = 0; i < d_size; i++){
		if ((density[i] < den_threshold) && (delta[i] > del_threshold)){
			Exc_data_index[(*exc_n)] = i;
			(*exc_n)++;
		}
	}
	*/
	//cout << "Find exception data starting...." << endl;
	int exc_num;
	int* exc = Find_Exc(density, delta, den_threshold, del_threshold, data_size, &exc_num);
	int* Exc_index = Find_Exc_plus(tem_core, exc, data_size, exc_num, exc_n);
	sort(Exc_index, Exc_index + *exc_n);
	for (int i = 0; i < *exc_n; i++){
		Exc_data_index[i] = Exc_index[i];
	}

	//cout << "Find exception data ending...." << endl;

	/*
	for (int i = 0; i < *exc_n; i++)
		std::cout << Exc_data_index[i] << ", ";
	std::cout << endl;
	*/
	delete exc; delete Exc_index; delete density;
	free(delta); free(tem_core);
}