
#define _CRT_SECURE_NO_WARNINGS

#include "stdio.h"
#include <stdlib.h>
#include <string.h>
#include <ctime>

//10000张mnist
#define COUNT_TEST		10000
#define FILE_TEST_IMAGE		"t10k-images-idx3-ubyte_withoutHeader"
#define FILE_TEST_LABEL		"t10k-labels-idx1-ubyte_withoutHeader"

//20张png
//#define COUNT_TEST		20
//#define FILE_TEST_IMAGE		"image-ubyte"
//#define FILE_TEST_LABEL		"label-ubyte"

typedef unsigned char uint8;
uint8 imageSet[COUNT_TEST][28][28];
uint8 labelSet[COUNT_TEST];




//weights
#define INPUT			1
#define LAYER1			6
#define LAYER2			6
#define LAYER3			16
#define LAYER4			16
#define LAYER5			128
#define LAYER6			84
#define LAYER7          10
#define LENGTH_KERNEL_0_1	5
#define LENGTH_KERNEL_2_3	5
#define LENGTH_KERNEL_4_5	4
#define LENGTH_KERNEL_5_6	1
#define LENGTH_KERNEL_6_7	1
double weight0_1[INPUT][LAYER1][LENGTH_KERNEL_0_1][LENGTH_KERNEL_0_1];
double weight2_3[LAYER2][LAYER3][LENGTH_KERNEL_2_3][LENGTH_KERNEL_2_3];
double weight4_5[LAYER4][LAYER5][LENGTH_KERNEL_4_5][LENGTH_KERNEL_4_5];
double weight5_6[LAYER5][LAYER6][LENGTH_KERNEL_5_6][LENGTH_KERNEL_5_6];
double weight6_7[LAYER6][LAYER7][LENGTH_KERNEL_6_7][LENGTH_KERNEL_6_7];
double bias0_1[LAYER1];
double bias2_3[LAYER3];
double bias4_5[LAYER5];
double bias5_6[LAYER6];
double bias6_7[LAYER7];



//feature maps
#define LENGTH_FEATURE0	28
#define LENGTH_FEATURE1	(LENGTH_FEATURE0 - LENGTH_KERNEL_0_1 + 1)  //24
#define LENGTH_FEATURE2	(LENGTH_FEATURE1 >> 1)                 //12
#define LENGTH_FEATURE3	(LENGTH_FEATURE2 - LENGTH_KERNEL_2_3 + 1)  //8
#define	LENGTH_FEATURE4	(LENGTH_FEATURE3 >> 1)			       //4
#define LENGTH_FEATURE5	(LENGTH_FEATURE4 - LENGTH_KERNEL_4_5+ 1)  //1
#define LENGTH_FEATURE6	1
#define LENGTH_FEATURE7	1
double input[INPUT][LENGTH_FEATURE0][LENGTH_FEATURE0];
double layer1[LAYER1][LENGTH_FEATURE1][LENGTH_FEATURE1];
double layer2[LAYER2][LENGTH_FEATURE2][LENGTH_FEATURE2];
double layer3[LAYER3][LENGTH_FEATURE3][LENGTH_FEATURE3];
double layer4[LAYER4][LENGTH_FEATURE4][LENGTH_FEATURE4];
double layer5[LAYER5][LENGTH_FEATURE5][LENGTH_FEATURE5];
double layer6[LAYER6][LENGTH_FEATURE6][LENGTH_FEATURE6];
double layer7[LAYER7][LENGTH_FEATURE7][LENGTH_FEATURE7];


#define GETLENGTH(array) (sizeof(array)/sizeof(*(array)))

#define GETCOUNT(array)  (sizeof(array)/sizeof(double))

#define FOREACH(i,count) for (int i = 0; i < count; ++i)


double relu(double x)
{
	return x * (x > 0);
}


#define CONVOLUTE_VALID(input,output,weight)											\
{																						\
	FOREACH(o0,GETLENGTH(output))														\
		FOREACH(o1,GETLENGTH(*(output)))												\
			FOREACH(w0,GETLENGTH(weight))												\
				FOREACH(w1,GETLENGTH(*(weight)))										\
					(output)[o0][o1] += (input)[o0 + w0][o1 + w1] * (weight)[w0][w1];	\
}

#define CONVOLUTION_FORWARD(input,output,weight,bias)							\
{																				\
	for (int x = 0; x < GETLENGTH(weight); ++x)									\
		for (int y = 0; y < GETLENGTH(*weight); ++y)							\
			CONVOLUTE_VALID(input[x], output[y], weight[x][y]);					\
	FOREACH(j, GETLENGTH(output))												\
		FOREACH(i, GETCOUNT(output[j]))											\
		((double *)output[j])[i] = relu(((double *)output[j])[i] + bias[j]);	\
}



#define SUBSAMP_MAX_FORWARD(input,output)														\
{																								\
	const int len0 = GETLENGTH(*(input)) / GETLENGTH(*(output));								\
	const int len1 = GETLENGTH(**(input)) / GETLENGTH(**(output));								\
	FOREACH(i, GETLENGTH(output))																\
	FOREACH(o0, GETLENGTH(*(output)))															\
	FOREACH(o1, GETLENGTH(**(output)))															\
	{																							\
		int x0 = 0, x1 = 0, ismax;																\
		FOREACH(l0, len0)																		\
			FOREACH(l1, len1)																	\
		{																						\
			ismax = input[i][o0*len0 + l0][o1*len1 + l1] > input[i][o0*len0 + x0][o1*len1 + x1];\
			x0 += ismax * (l0 - x0);															\
			x1 += ismax * (l1 - x1);															\
		}																						\
		output[i][o0][o1] = input[i][o0*len0 + x0][o1*len1 + x1];								\
	}																							\
}

uint8 Predict()
{
	memset(layer1, 0, LAYER1 * LENGTH_FEATURE1 * LENGTH_FEATURE1 * sizeof(double));
	memset(layer2, 0, LAYER2 * LENGTH_FEATURE2 * LENGTH_FEATURE2 * sizeof(double));
	memset(layer3, 0, LAYER3 * LENGTH_FEATURE3 * LENGTH_FEATURE3 * sizeof(double));
	memset(layer4, 0, LAYER4 * LENGTH_FEATURE4 * LENGTH_FEATURE4 * sizeof(double));
	memset(layer5, 0, LAYER5 * LENGTH_FEATURE5 * LENGTH_FEATURE5 * sizeof(double));
	memset(layer6, 0, LAYER6 * LENGTH_FEATURE6 * LENGTH_FEATURE6 * sizeof(double));
	memset(layer7, 0, LAYER7 * LENGTH_FEATURE7 * LENGTH_FEATURE7 * sizeof(double));

	CONVOLUTION_FORWARD(input, layer1, weight0_1, bias0_1);
	SUBSAMP_MAX_FORWARD(layer1, layer2);
	CONVOLUTION_FORWARD(layer2, layer3, weight2_3, bias2_3);
	SUBSAMP_MAX_FORWARD(layer3, layer4);
	CONVOLUTION_FORWARD(layer4, layer5, weight4_5, bias4_5);
	CONVOLUTION_FORWARD(layer5, layer6, weight5_6, bias5_6);
	CONVOLUTION_FORWARD(layer6, layer7, weight6_7, bias6_7);

	int ans = 0;
	double maxValue = layer7[ans][0][0];
	for (int i = 1; i < 10; i++) {
		if (layer7[i][0][0] > maxValue) {
			maxValue = layer7[i][0][0];
			ans = i;
		}
	}
	return ans;
}






int read_data(const char data_file[], const char label_file[])
{
	FILE* fp_image = fopen(data_file, "rb");
	FILE* fp_label = fopen(label_file, "rb");
	if (!fp_image || !fp_label) return 1;
	fseek(fp_image, 0, SEEK_SET);
	fseek(fp_label, 0, SEEK_SET);
	fread(imageSet, sizeof(*imageSet) * COUNT_TEST, 1, fp_image);
	fread(labelSet, COUNT_TEST, 1, fp_label);
	fclose(fp_image);
	fclose(fp_label);
	return 0;
}



#define readWeightMat(FILE_MAT, KERNELSIZE, frontLayerSize,w)	\
{																\
	FILE* f = fopen(FILE_MAT, "r");								\
	if (!f) return 1;											\
	double tmp = 0;												\
	int idx = 0;												\
	while (fscanf(f, "%lf", &tmp) != EOF) {						\
		w[idx / KERNELSIZE / KERNELSIZE % frontLayerSize][idx / KERNELSIZE / KERNELSIZE / frontLayerSize][idx / KERNELSIZE % KERNELSIZE][idx % KERNELSIZE] = tmp; \
		idx++;												    \
	}														    \
	fclose(f);												    \
}  

#define readBias(FILE_MAT, w)									\
{																\
	FILE* f = fopen(FILE_MAT, "r");								\
	if (!f) return 1;											\
	double tmp = 0;												\
	int idx = 0;												\
	while (fscanf(f, "%lf", &tmp) != EOF) {						\
		w[idx++] = tmp;									        \
	}															\
	fclose(f);													\
}  

int main()
{
	//load test image and label
	if (read_data(FILE_TEST_IMAGE, FILE_TEST_LABEL))
	{
		printf("ERROR!!!\nDataset File Not Find!Please Copy Dataset to the Floder Included the exe\n");  
	}
	//load weights
	//load c1
	readWeightMat("w0_1",LENGTH_KERNEL_0_1,INPUT,weight0_1);
	//load bias0_1
	readBias("bias0_1",bias0_1);
   
	//load c2
	readWeightMat("w2_3",LENGTH_KERNEL_2_3,LAYER1,weight2_3);
	//load bias2_3
	readBias("bias2_3", bias2_3);

	//load d1
	readWeightMat("w4_5", LENGTH_KERNEL_4_5, LAYER4, weight4_5);
	//load bias4_5
	readBias("bias4_5", bias4_5);

	//load d2
	readWeightMat("w5_6", LENGTH_KERNEL_5_6, LAYER5, weight5_6);
	//load bias5_6
	readBias("bias5_6", bias5_6);

	//load d3
	readWeightMat("w6_7", LENGTH_KERNEL_6_7, LAYER6, weight6_7);
	//load bias6_7
	readBias("bias6_7", bias6_7);

	int right = 0;
	for (int i = 0; i < COUNT_TEST; ++i)
	{
		uint8 ans = labelSet[i];
		for (int u = 0; u < LENGTH_FEATURE0; u++) {
			for (int v = 0; v < LENGTH_FEATURE0; v++) {
				//input[0][u][v] = 255-imageSet[i][u][v];  //反相
				//input[0][u][v] = imageSet[i][u][LENGTH_FEATURE0 - v -1];   //轴对称
				input[0][u][v] = imageSet[i][u][v ];   
			}
		}
		int p = Predict();
		//printf("predict:%d  fact:%d\n", p, ans);
		if (p == ans) {
			right++;
		}
	}
	
	printf("%d correct out of %d samples\n",right,COUNT_TEST);


	return 0;
}

