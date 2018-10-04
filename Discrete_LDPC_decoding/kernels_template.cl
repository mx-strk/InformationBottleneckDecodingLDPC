//#pragma OPENCL EXTENSION cl_khr_fp64: enable
#define CN_DEGREE ${ cn_degree }
#define VN_DEGREE ${ vn_degree }

void kernel test_kernel()
{
  int gid=get_global_id(0);
  int lid=get_local_id(0);

  
}
 
void kernel send_channel_values_to_checknode_inbox(global const int* channel_values,
						   global const int* inbox_memory_start_varnodes,
						   global const int* degree_varnode_nr, 
						   global const int* target_memorycells_varnodes,
						   global int* checknode_inbox
						  )
{
  int gid1=get_global_id(0);
  int gid2=get_global_id(1);


  for(int w=0;w!=degree_varnode_nr[gid1];w++)
  {
    int target_memcell=target_memorycells_varnodes[inbox_memory_start_varnodes[gid1]+w];
    checknode_inbox[target_memcell*${ msg_at_time }+gid2]=channel_values[gid1*${ msg_at_time }+gid2];
  }

}

void kernel checknode_update_iter0(global const int* checknode_inbox,
				   global const int* inbox_memory_start_checknodes,
				   global const int* degree_checknode_nr, 
				   global const int* target_memorycells_checknodes,
				   global int* varnode_inbox,
				   int cardinality_T_channel,
				   int cardinality_T_decoder_ops,
				   global const int* Trellis_checknodevector_a				   
				  )

{
  int gid1=get_global_id(0);
  int gid2=get_global_id(1);

  int lid=get_local_id(0);
  int local_size=get_local_size(0);

  private int current_msgs[CN_DEGREE];
  
  int node_degree = degree_checknode_nr[gid1];

  /*int size_of_lookup_vector_this_iter=cardinality_T_channel*cardinality_T_channel
      +(node_degree-3)*cardinality_T_decoder_ops*cardinality_T_channel;
  */

  //read lookup vector for the current operation to the local memory
  
  int msgs_start_idx=inbox_memory_start_checknodes[gid1];
  
  
  for (int w=0;w!=node_degree;w++)
  {
  int target_memcell=target_memorycells_checknodes[inbox_memory_start_checknodes[gid1]+w];
  
  
  int pos=0;
  for(int v=0;v!=node_degree;v++)
  {
    if(v!=w) {


      current_msgs[pos]=checknode_inbox[ (msgs_start_idx+v)*${ msg_at_time }+gid2];
      pos++;
    }  
  }
  
  int t_0=Trellis_checknodevector_a[current_msgs[0]*cardinality_T_channel+current_msgs[1]];

  int t_lm1=t_0;
  for(int l=1;l<node_degree-2;l++)
  {
    t_lm1=Trellis_checknodevector_a[t_lm1*cardinality_T_decoder_ops+current_msgs[l+1]+
			      cardinality_T_channel*cardinality_T_channel+
			      (l-1)*cardinality_T_channel*cardinality_T_decoder_ops];   

  }
  varnode_inbox[target_memcell*${ msg_at_time }+gid2]=t_lm1;
  }
}



void kernel varnode_update(global const int* channel_values,
			   global const int* varnode_inbox,
			   global const int* inbox_memory_start_varnodes,
		       global const int* degree_varnode_nr,
			   global const int* target_memorycells_varnodes,
		       global int* checknode_inbox,
		       int cardinality_T_channel,
			   int cardinality_T_decoder_ops,
			   int iteration,
			   global const int* Trellis_varnodevector_a)

{
  int gid1=get_global_id(0);
  int gid2=get_global_id(1);


  int lid=get_local_id(0);
  int local_size=get_local_size(0);
  int node_degree=degree_varnode_nr[gid1];
  
  private int current_msgs[VN_DEGREE];


  int size_of_lookup_vector_this_iter=cardinality_T_channel*cardinality_T_decoder_ops+(node_degree-1)*cardinality_T_decoder_ops*cardinality_T_decoder_ops;
  //printf("%d ",size_of_lookup_vector_this_iter);
  int offset=iteration*size_of_lookup_vector_this_iter;

  //read lookup vector for the current operation to the local memory
  local int lcl_trellis_relevant[768];

  for(int k=lid;k<size_of_lookup_vector_this_iter;k+=local_size)
  {
    lcl_trellis_relevant[k]=Trellis_varnodevector_a[offset+k];
    //printf("%d ",lcl_trellis_relevant[k]);
  }

  barrier(CLK_LOCAL_MEM_FENCE);


  int msgs_start_idx=inbox_memory_start_varnodes[gid1];

  current_msgs[0]=channel_values[gid1*${ msg_at_time }+gid2];

  for (int w=0;w!=node_degree;w++)
  {
  int target_memcell=target_memorycells_varnodes[inbox_memory_start_varnodes[gid1]+w];
  

  int pos=1;
  for(int v=0;v!=node_degree;v++)
  {
    if(v!=w){
        current_msgs[pos]=varnode_inbox[(msgs_start_idx+v)*${ msg_at_time }+gid2];
        pos=pos+1;
    }  
  }
  
  int t_0=Trellis_varnodevector_a[offset+current_msgs[0]*cardinality_T_decoder_ops+current_msgs[1]];
  //int t_0=lcl_trellis_relevant[current_msgs[0]*cardinality_T_decoder_ops+current_msgs[1]];

  int t_lm1=t_0;
  for(int l=1;l<=node_degree-2;l++)
  {

  /*t_lm1=Trellis_varnodevector_a[offset+t_lm1*cardinality_T_decoder_ops+current_msgs[l+1]+
			       cardinality_T_channel*cardinality_T_decoder_ops+
			       (l-1)*cardinality_T_decoder_ops*cardinality_T_decoder_ops];*/
  t_lm1=lcl_trellis_relevant[t_lm1*cardinality_T_decoder_ops+current_msgs[l+1]+
			       cardinality_T_channel*cardinality_T_decoder_ops+
			       (l-1)*cardinality_T_decoder_ops*cardinality_T_decoder_ops];

 
  }
  checknode_inbox[target_memcell*${ msg_at_time }+gid2]=t_lm1;
  
  }
  
}

void kernel checknode_update(global const int* checknode_inbox,
				   global const int* inbox_memory_start_checknodes,
				   global const int* degree_checknode_nr, 
				   global const int* target_memorycells_checknodes,
				   global int* varnode_inbox,
				   int cardinality_T_channel,
				   int cardinality_T_decoder_ops,
				   int iteration,
				   global const int* Trellis_checknodevector_a
				  )

{
  int gid1=get_global_id(0);
  int gid2=get_global_id(1);


  int lid=get_local_id(0);
  int local_size=get_local_size(0);

  int node_degree=degree_checknode_nr[gid1];

  private int current_msgs[CN_DEGREE-1];

  //int size_of_lookup_vector_this_iter=cardinality_T_decoder_ops*cardinality_T_decoder_ops*(node_degree-2);

  //read lookup vector for the current operation to the local memory
  int offset=cardinality_T_channel*cardinality_T_channel+(node_degree-3)*cardinality_T_channel*cardinality_T_decoder_ops+
	     iteration*((node_degree-2)*cardinality_T_decoder_ops*cardinality_T_decoder_ops);

  /*local int lcl_trellis_relevant[1024];

  for(int k=lid;k<size_of_lookup_vector_this_iter;k+=local_size)
  {
    lcl_trellis_relevant[k]=Trellis_checknodevector_a[offset+k];
    //printf("%d ",lcl_trellis_relevant[k]);
  }

  barrier(CLK_LOCAL_MEM_FENCE); */


  int msgs_start_idx=inbox_memory_start_checknodes[gid1];

  for (int w=0;w!=node_degree;w++)
  {
  int target_memcell=target_memorycells_checknodes[inbox_memory_start_checknodes[gid1]+w];

  int pos=0;
  for(int v=0;v!=node_degree;v++)
  {
    if(v!=w) current_msgs[pos++]=checknode_inbox[(msgs_start_idx+v)*${ msg_at_time }+gid2];
  }


  int t_lm1=current_msgs[0];

  for(int l=0;l<node_degree-2;l++)
  {
    t_lm1=Trellis_checknodevector_a[offset+t_lm1*cardinality_T_decoder_ops+current_msgs[l+1]+
			      (l)*cardinality_T_decoder_ops*cardinality_T_decoder_ops];
	/*t_lm1=lcl_trellis_relevant[offset+t_lm1*cardinality_T_decoder_ops+current_msgs[l+1]+
			      (l)*cardinality_T_decoder_ops*cardinality_T_decoder_ops];*/

  }
  varnode_inbox[target_memcell*${ msg_at_time }+gid2]=t_lm1;
  }
}


void kernel calc_varnode_output(global const int* channel_values,
			   global const int* varnode_inbox,
			   global const int* inbox_memory_start_varnodes,
		       global const int* degree_varnode_nr,
		       int cardinality_T_channel,
			   int cardinality_T_decoder_ops,
			   int iteration,
			   global const int* Trellis_varnodevector_a,
		       global int* varnode_output)

{
  int gid1=get_global_id(0);
  int gid2=get_global_id(1);

  int lid=get_local_id(0);
  int local_size=get_local_size(0);

  //printf("%d ,",gid);

  int node_degree=degree_varnode_nr[gid1];

  private int current_msgs[VN_DEGREE+1];

  int size_of_lookup_vector_this_iter=cardinality_T_channel*cardinality_T_decoder_ops+
      +(node_degree-1)*cardinality_T_decoder_ops*cardinality_T_decoder_ops;

  int offset=iteration*size_of_lookup_vector_this_iter;


  int msgs_start_idx=inbox_memory_start_varnodes[gid1];
  current_msgs[0]=channel_values[gid1*${ msg_at_time }+gid2];


  for(int v=0;v!=node_degree;v++)
  {
    current_msgs[v+1]=varnode_inbox[(msgs_start_idx+v)*${ msg_at_time }+gid2];
  }

  int t_0=Trellis_varnodevector_a[offset+current_msgs[0]*cardinality_T_decoder_ops+current_msgs[1]];
  int t_lm1=t_0;
  for(int l=1;l!=node_degree;l++)
  {
    t_lm1=Trellis_varnodevector_a[offset+t_lm1*cardinality_T_decoder_ops+current_msgs[l+1]+
			       cardinality_T_channel*cardinality_T_decoder_ops+
			       (l-1)*cardinality_T_decoder_ops*cardinality_T_decoder_ops];

  }
  varnode_output[gid1*${ msg_at_time }+gid2]=(int) (t_lm1);

}

void kernel calc_syndrome(global const int* checknode_inbox,
			  global const int* inbox_memory_start_checknodes,
			  global const int* degree_checknode_nr,
			  int cardinality_T_decoder_ops,
			  global int* syndrome)
{
  int gid1=get_global_id(0);
  int gid2=get_global_id(1);

  int lid=get_local_id(0);

  int node_degree=degree_checknode_nr[gid1];
  int start_idx=inbox_memory_start_checknodes[gid1];


  int mod_sum=0;
  for (int w=0;w!=node_degree;w++)
  {
    mod_sum+=(int)( checknode_inbox[(start_idx+w)*${ msg_at_time }+gid2 ]<cardinality_T_decoder_ops/2);
    mod_sum=mod_sum%2;
  }
  syndrome[gid1*${ msg_at_time }+gid2]=mod_sum;
}