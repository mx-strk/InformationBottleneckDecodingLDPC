
__kernel void quantize(const int cardinality_T,
              global const double* channel_values,
			  global const double* limits,
			  global int* clusters)
{
  int gid0 = get_global_id(0);
  int gid1 = get_global_id(1);

  int cluster_val;
  cluster_val = 0;
  int Nvar = ${Nvar} ;
  //double lim[17];

    //for (int k = 0; k < cardinality_T; k++)
    // lim[k] = limits[k];

    for (int w=1;w!=cardinality_T;w++)
    {
         if (( channel_values[gid0+gid1*Nvar]-limits[w]) > 0) {
            cluster_val = cluster_val + 1;
         }

    }
  clusters[gid0+gid1*Nvar] = cluster_val;

}

__kernel void quantize_LLR(const int cardinality_T,
              global const double* channel_values,
			  global const double* limits,
			  global const double* LLR_vector,
			  global double* LLRs)
{
  int gid0 = get_global_id(0);
  int gid1 = get_global_id(1);

  int cluster_val;
  cluster_val = 0;
  int Nvar = ${Nvar} ;


    for (int w=1;w!=cardinality_T;w++)
    {
         if (( channel_values[gid0+gid1*Nvar]-limits[w]) > 0) {
            cluster_val = cluster_val + 1;
         }

    }
  LLRs[gid0+gid1*Nvar] = LLR_vector[cluster_val];

}