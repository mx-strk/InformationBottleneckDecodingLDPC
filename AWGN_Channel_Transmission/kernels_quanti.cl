
__kernel void quantize(const int cardinality_T,
              global const float* channel_values,
			  global const float* limits,
			  global int* clusters)
{
  int gid0 = get_global_id(0);
  int gid1 = get_global_id(1);

  int cluster_val;
  cluster_val = 0;
  //float lim[16];

    //for (int k = 0; k < cardinality_T; k++)
    //lim[k] = limits[k];

    for (int w=1;w!=cardinality_T;w++)
    {
         if (( channel_values[gid0+gid1*10000]-limits[w]) > 0) {
            cluster_val = cluster_val + 1;
         }

    }
  clusters[gid0+gid1*10000] = cluster_val;

}