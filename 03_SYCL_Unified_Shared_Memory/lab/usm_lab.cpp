//==============================================================
// Copyright © Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <sycl/sycl.hpp>
#include <cmath>
using namespace sycl;

static const int N = 1024;

int main() {
  queue q;
  std::cout << "Device : " << q.get_device().get_info<info::device::name>() << "\n";

  //intialize 2 arrays on host
  int *data1 = static_cast<int *>(malloc(N * sizeof(int)));
  int *data2 = static_cast<int *>(malloc(N * sizeof(int)));
  for (int i = 0; i < N; i++) {
    data1[i] = 25;
    data2[i] = 49;
  }

  // Part I: for both EECE.4510 and EECE.5510 students
  //  
  //# STEP 1 : Create USM device allocation for data1 and data2

  int *data1_device = malloc_device<int>(N, q);
  int *data2_device = malloc_device<int>(N, q);





    
  //# STEP 2 : Copy data1 and data2 to USM device allocation
    
  q.memcpy(data1_device, data1, sizeof(int) * N).wait();  // Need to copy ENTIRE size of array, not just single element
  q.memcpy(data2_device, data2, sizeof(int) * N).wait();  // Need to copy ENTIRE size of array, not just single element





  //# STEP 3 : Write kernel code to update data1 on device with square of its value

  q.parallel_for(N, [=](auto i) { 

    data1_device[i] = std::sqrt(data1_device[i]);  // sqrt() comes from the std namespace
      
  });
  q.wait();  // wait() needed to synchronize



  //# STEP 4 : Write kernel code to update data2 on device with square of its value
    
  q.parallel_for(N, [=](auto i) { 

    data2_device[i] = std::sqrt(data2_device[i]);  // sqrt() comes from the std namespace
      
  });
  q.wait();  // wait() needed to synchronize



  //# STEP 5 : Write kernel code to add data2 on device to data1
    
  q.parallel_for(N, [=](auto i) { 

    data1_device[i] += data2_device[i];
      
  });
  q.wait();  // wait() needed to synchronize



  //# STEP 6 : Copy data1 on device to host
    
  q.memcpy(data1, data1_device, sizeof(int) * N).wait();  // Need to copy ENTIRE size of array, not just single element



  //# verify results
  int fail = 0;
  for (int i = 0; i < N; i++) if(data1[i] != 12) {fail = 1; break;}
  if(fail == 1) std::cout << " FAIL"; else std::cout << " PASS";
  std::cout << "\n";


  //# STEP 7 : Free USM device allocations and other allocated memory on host

  free(data1_device, q);
  free(data2_device, q);


  // END OF PART I
  
  // Part II. for EECE.5510 students only
  //
  // Add event based kernel dependency for the Steps 2 - 6
  // Hint: Since you have "freed" all the memory used in previous steps,
  //       you need to re-initialize and re-allocate them. Remember to free 
  //       them at the end. Verification of the results is also needed. 

  //Reinitialize 2 arrays on host
  for (int i = 0; i < N; i++) {
    data1[i] = 25;
    data2[i] = 49;
  }

  //# STEP 1b : Create USM device allocation for data1 and data2

  data1_device = malloc_device<int>(N, q);  // No need to re-declare
  data2_device = malloc_device<int>(N, q);  // No need to re-declare

  //# STEP 2b : Copy data1 and data2 to USM device allocation
    
  event e_copy1 = q.memcpy(data1_device, data1, sizeof(int) * N);  // Using event-based depenedency management instead of wait()
  event e_copy2 = q.memcpy(data2_device, data2, sizeof(int) * N);  // Using event-based depenedency management instead of wait()

  //# STEP 3b : Write kernel code to update data1 on device with square of its value

  event e_sqrt1 = q.parallel_for(N, {e_copy1}, [=](auto i) { 
    
    data1_device[i] = std::sqrt(data1_device[i]);
      
  });

  //# STEP 4b : Write kernel code to update data2 on device with square of its value

  event e_sqrt2 = q.parallel_for(N, {e_copy2}, [=](auto i) { 

    data2_device[i] = std::sqrt(data2_device[i]);
      
  });

  //# STEP 5b : Write kernel code to add data2 on device to data1
    
  q.parallel_for(N, {e_sqrt1, e_sqrt2}, [=](auto i) { 

    data1_device[i] += data2_device[i];
      
  });

  //# STEP 6b : Copy data1 on device to host
    
  q.memcpy(data1, data1_device, sizeof(int) * N).wait();

  //# verify results
  fail = 0;  // No need to re-declare
  for (int i = 0; i < N; i++) if(data1[i] != 12) {fail = 1; break;}
  if(fail == 1) std::cout << " FAIL"; else std::cout << " PASS";
  std::cout << "\n";


  //# STEP 7b : Free USM device allocations and other allocated memory on host

  free(data1_device, q);
  free(data2_device, q);

  return 0;
}

