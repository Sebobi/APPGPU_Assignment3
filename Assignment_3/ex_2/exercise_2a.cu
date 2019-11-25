
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include <cstdlib>
#include <ctime>
#include <chrono>
#include <cmath>

struct Particle {
	float3 position;
	float3 velocity;
};

bool compareParticles(Particle *particles1, Particle *particles2, int size);

void updateParticlesCPU(Particle *particles, int size, float velocityGiven);

void initializeParticles(Particle *particles, int arraySize);

__global__ void UPDATE_PARTICLES(Particle *particles, const float velocityGiven, const int arraySize)
{

	int i = blockDim.x*blockIdx.x + threadIdx.x;
	if (i < arraySize) {
		particles[i].velocity.x = velocityGiven * (i + 1);
		particles[i].position.x = particles[i].position.x + particles[i].velocity.x * 1;
	}

}

int main()
{
	const int NUM_PARTICLES = 40000;
	const int NUM_ITERATIONS = 1000;
	const int BLOCK_SIZE = 256;

	const int BLOCKS = (NUM_PARTICLES + BLOCK_SIZE - 1) / BLOCK_SIZE;

	Particle particles[NUM_PARTICLES] = {};


	const float randomVelocity = rand();
	float d2 = rand();

	printf("%d\n", fabs(0.2) < fabs(0.5) ? 1 : 0);


	initializeParticles(particles, NUM_PARTICLES);

	Particle *particles_GPU = 0;
	cudaError_t cudaStatus;

	printf("Starting GPU particle simulation now\n");

	//Time before GPU runs update
	auto current_time = std::chrono::system_clock::now();
	auto duration_in_seconds = std::chrono::duration<double>(current_time.time_since_epoch());
	double gpu_before = duration_in_seconds.count();


	//cudaStatus = cudaMalloc((void**)&particles_GPU, NUM_PARTICLES * sizeof(Particle));
	//cudaStatus = cudaMallocHost((void**)&particles_GPU, NUM_PARTICLES * sizeof(Particle));
	cudaStatus = cudaMallocManaged(&particles_GPU, NUM_PARTICLES * sizeof(Particle));

	cudaStatus = cudaMemcpy(particles_GPU, particles, NUM_PARTICLES * sizeof(Particle), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		printf("malloc failed\n");
		return 1;
	}


	//TIMESTEP
	for (int i = 0; i < NUM_ITERATIONS; i++) {

		//Update
		UPDATE_PARTICLES << <BLOCKS, BLOCK_SIZE >> > (particles_GPU, randomVelocity, NUM_PARTICLES);

		cudaDeviceSynchronize();

		//work with data here
		Particle temp  = particles_GPU[1];
		particles[1] = particles[0];
		particles[0] = temp;
		temp = particles[1];
		particles[1] = particles[0];
		particles[0] = temp;

	}

	cudaDeviceSynchronize();

	if (cudaStatus != cudaSuccess)
	{
		cudaFreeHost(particles_GPU);
		return 1;
	}

	current_time = std::chrono::system_clock::now();
	duration_in_seconds = std::chrono::duration<double>(current_time.time_since_epoch());
	double gpu_after = duration_in_seconds.count();

	printf("GPU Finished in time:%f\n", (gpu_after - gpu_before));


	current_time = std::chrono::system_clock::now();
	duration_in_seconds = std::chrono::duration<double>(current_time.time_since_epoch());
	double CPU_before = duration_in_seconds.count();

	for (int i = 0; i < NUM_ITERATIONS; i++) {
		updateParticlesCPU(particles, NUM_PARTICLES, randomVelocity);
	}

	current_time = std::chrono::system_clock::now();
	duration_in_seconds = std::chrono::duration<double>(current_time.time_since_epoch());
	double CPU_after = duration_in_seconds.count();

	printf("CPU Finished in time:%f\n", (CPU_after - CPU_before));


	printf("Comparing particles...");

	bool result = compareParticles(particles_GPU, particles, NUM_PARTICLES);
	if (result == true) {
		printf(" Correct!\n");
	}
	else {
		printf(" Not the same...\n");
	}

	cudaFree(particles_GPU);


	return 0;

}




bool compareParticles(Particle *particles1, Particle *particles2, int size) {

	for (int i = 0; i < size; i++) {
		if (fabs(particles1[i].position.x - particles2[i].position.x) > 0.01 || fabs(particles1[i].velocity.x - particles2[i].velocity.x) > 0.01)
			return false;
	}
	return true;
}


void updateParticlesCPU(Particle *particles, int size, float velocityGiven) {
	for (int i = 0; i < size; i++) {
		particles[i].velocity.x = velocityGiven * (i + 1);
		particles[i].position.x = particles[i].position.x + particles[i].velocity.x * 1;
	}


}



void initializeParticles(Particle *particles, int arraySize) {

	for (int i = 0; i < arraySize; i++) {
		particles[i].velocity.x = rand();
		particles[i].velocity.y = rand();
		particles[i].velocity.z = rand();
		particles[i].position.x = rand();
		particles[i].position.y = rand();
		particles[i].position.z = rand();
	}

}
