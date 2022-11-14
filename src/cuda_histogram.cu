#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp> // cv::imread
#include <opencv2/imgproc.hpp> // cv::calcHist

#include <chrono> // std::chrono::steady_clock
#include <iostream>
#include <filesystem>
#include <fstream> // std::ofstream

namespace fs = std::filesystem;

#define checkCudaErrors(call)                                 \
  do {                                                        \
    cudaError_t err = call;                                   \
    if (err != cudaSuccess) {                                 \
      std::cerr << "CUDA error at " << __FILE__ << " " << __LINE__ << ": " << cudaGetErrorString(err) << "\n"; \
      exit(EXIT_FAILURE);                                     \
    }                                                         \
  } while (0)

// Kernel constants corresponding to the input image
constexpr int NUM_BINS = 256;
constexpr int NUM_BLOCK_X = 64;
constexpr int NUM_BLOCK_Y = 64;
constexpr int NUM_PARTS = NUM_BLOCK_X * NUM_BLOCK_Y;
constexpr int NUM_THREAD_IN_BLOCK_X = 16;
constexpr int NUM_THREAD_IN_BLOCK_Y = 16;

// Display some device properties
void displayCudaDeviceProperties(const cudaDeviceProp& prop)
{
  std::cout << "+ Device " << prop.name << "\n";
  std::cout << "\t* Memory clock rate " << prop.memoryClockRate * 1e-6 << " GHz\n";
  std::cout << "\t* Shared memory per block " << prop.sharedMemPerBlock * 1e-3 << " Mega Bytes\n";
  std::cout << "\t* Global memory " << prop.totalGlobalMem * 1e-9 << " Giga Bytes\n";
  std::cout << "\t* Device compute capability " << prop.major << "." << prop.minor << "\n";
  std::cout << "\t* Max threads per dimension x: " << prop.maxThreadsDim[0] << " y: " << prop.maxThreadsDim[1] << " z: " << prop.maxThreadsDim[2] << "\n";
  std::cout << "\t* Max threads per block " << prop.maxThreadsPerBlock << "\n";
  std::cout << "\t* Max threads per SM " << prop.maxThreadsPerMultiProcessor << "\n";
  std::cout << "\t* Number of SM " << prop.multiProcessorCount << "\n";
}

// Display chuncks of memory (debug purposes)
template<typename T>
void displayHistUpTo(const T* hist, std::size_t up_to)
{
  for(auto i=0u; i < up_to; ++i) {
    std::cout << hist[i] << " ";
  }
  std::cout << std::endl;
}

// Allocate and copy the input image, allocate the destination histogram
std::tuple<unsigned char*, unsigned int*, unsigned int*> allocateDeviceMemory(const unsigned char* h_image, int rows, int cols)
{
  unsigned char* d_image = nullptr;
  unsigned int* d_temp_histograms = nullptr;
  unsigned int* d_final_histogram = nullptr;
  // Copy image to global memory
  const auto size = rows * cols * sizeof(unsigned char);
  checkCudaErrors(cudaMalloc(&d_image, size));
  checkCudaErrors(cudaMalloc(&d_temp_histograms, sizeof(unsigned int) * NUM_PARTS * NUM_BINS));
  checkCudaErrors(cudaMalloc(&d_final_histogram, sizeof(unsigned int) * NUM_BINS));
  return {d_image, d_temp_histograms, d_final_histogram};
}

__global__ void accumulateLocalHistograms(const unsigned char* d_image, int width, int height, unsigned int* global_mem)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  int block_id = blockIdx.y * gridDim.x + blockIdx.x;
  int thread_id_in_block = threadIdx.y * blockDim.x + threadIdx.x;

  // Initialize shared memory
  __shared__ unsigned int shared_hist[NUM_BINS];
  shared_hist[thread_id_in_block] = 0;
  __syncthreads();


  // Count intensity values in the local histogram
  const auto intensity = d_image[y * width + x];
  atomicAdd(&shared_hist[intensity], 1);
  __syncthreads();

  // Copy local histogram to global memory
  global_mem += block_id * NUM_BINS;

  global_mem[thread_id_in_block] = shared_hist[thread_id_in_block];
}


__global__ void accumulateFinalHistogram(unsigned int* d_temp_histograms, unsigned int* final_histogram)
{
  int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
  if(thread_id < NUM_BINS)
  {
    auto total{0u};
    for(auto i=0; i < NUM_PARTS; ++i) {
      total += d_temp_histograms[i * NUM_BINS + thread_id];
    }
    final_histogram[thread_id] = total;
  }
}

void resetDevice()
{
  checkCudaErrors(cudaDeviceReset());
}


__host__ std::vector<unsigned int> calcHistogramGPU(unsigned char* d_image, unsigned int* d_temp_histograms, unsigned int* d_final_histogram, int rows, int cols)
{
  const dim3 grid_dim(NUM_BLOCK_X, NUM_BLOCK_Y, 1);
  const dim3 block_dim(NUM_THREAD_IN_BLOCK_X, NUM_THREAD_IN_BLOCK_Y, 1);
  // Compute local histograms
  accumulateLocalHistograms<<<grid_dim, block_dim>>>(d_image, cols, rows, d_temp_histograms);
  // Accumulate final histogram
  accumulateFinalHistogram<<<NUM_PARTS, NUM_BINS>>>(d_temp_histograms, d_final_histogram);

  // Copy the final histogram to host array
  std::vector<unsigned int> hist_gpu(NUM_BINS, 0);
  checkCudaErrors(cudaMemcpy(hist_gpu.data(), d_final_histogram, sizeof(unsigned int) * NUM_BINS, cudaMemcpyDeviceToHost));
  return hist_gpu;
}

__host__ cv::Mat calcHistogramCPU(const cv::Mat& image)
{
  const float hrange[]= {0, 256};
  const float* hist_range[] = {hrange};
  const bool uniform = true;
  const bool accumulate = false;
  const int channels = 0;
  // Underlying hist type is float
  cv::Mat hist;
  cv::calcHist(&image, 1, &channels, cv::Mat(), hist, 1, &NUM_BINS, hist_range, uniform, accumulate);
  return hist;  
}

void writeResultHistograms(const cv::Mat& hist_cpu, const std::vector<unsigned int>& hist_gpu, const std::string& filename)
{
  std::ofstream res(filename, std::ios::trunc | std::ios::out);
  if(not res.is_open()) {
    std::cerr << "- Cannot open result file\n";
    return;
  }

  const auto write_hist = [&res](const auto* p, const auto num_bins)
  {
    for(auto i=0; i < num_bins; ++i) {
      res << p[i];
      if(i != num_bins - 1)
        res << ",";
    }
    res << std::endl;
  };
  write_hist(hist_cpu.ptr<float>(), NUM_BINS);
  write_hist(hist_gpu.data(), NUM_BINS);
  
  auto* p_gpu = hist_gpu.data();
  auto* p_cpu = hist_cpu.ptr<float>();
  for(auto i=0u; i < NUM_BINS; ++i) {
    res << std::abs(p_gpu[i] - p_cpu[i]);
    if(i != NUM_BINS - 1)
      res << ",";
  }
}


std::vector<std::pair<std::string, cv::Mat>> readImages(const fs::path& directory)
{
  std::vector<std::pair<std::string, cv::Mat>> images;
  for(const auto& filenames : std::filesystem::directory_iterator{directory})
  {
    images.emplace_back(filenames.path().stem(), cv::imread(filenames.path().string(), cv::IMREAD_GRAYSCALE));
  }
  return images;
}

int main(int argc, char** argv)
{
  int count;
  checkCudaErrors(cudaGetDeviceCount(&count));
  if(not count) {
    std::cerr << "- No cuda device found\n";
    return -1;
  }
  if(argc != 2) {
    std::cerr << "- Please provide the path to an image file\n";
    return -1;
  }

  // 1. Display cuda device information
  std::cout << "+ Found " << count << " devices\n";
  int device_id;
  checkCudaErrors(cudaGetDevice(&device_id));
  auto prop = cudaDeviceProp{};
  checkCudaErrors(cudaGetDeviceProperties(&prop, device_id));
  displayCudaDeviceProperties(prop);

  // 2. Read and allocate memory for the input image
  const auto images = readImages(argv[1]);
  if(std::empty(images)) {
    std::cerr << "- Could not find any images in " << argv[1] << "\n";
    return -1;
  }

  auto [d_image, d_global_histograms, d_final_histogram] = allocateDeviceMemory(images.front().second.ptr<unsigned char>(), images.front().second.rows, images.front().second.cols);
  for(const auto& [filename, image] : images)
  {
    const auto rows = image.rows;
    const auto cols = image.cols;
    const auto image_size = rows * cols * sizeof(unsigned char);
    auto now = std::chrono::steady_clock::now();
    checkCudaErrors(cudaMemcpy(d_image, image.ptr<unsigned char>(), image_size, cudaMemcpyHostToDevice));
    // 3. Run the kernels
    const auto hist_gpu = calcHistogramGPU(d_image, d_global_histograms, d_final_histogram, image.rows, image.cols);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - now).count();
    std::cout << "+ GPU histogram took " << ms << " ms\n";

    // 4. Calculate histogram using CPU opencv code
    now = std::chrono::steady_clock::now();
    const auto hist_cpu = calcHistogramCPU(image);
    ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - now).count();
    std::cout << "+ CPU histogram took " << ms << " ms\n";
  
    // 5. Write cpu and gpu histogram as well as their differences to a results.csv file
    writeResultHistograms(hist_cpu, hist_gpu, filename + ".csv");
  }

  // 6. Free device memory and device clean up
  checkCudaErrors(cudaFree(d_image));
  checkCudaErrors(cudaFree(d_global_histograms));
  checkCudaErrors(cudaFree(d_final_histogram));
  resetDevice();
  return 0;
}
