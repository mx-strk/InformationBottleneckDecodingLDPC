#define CN_DEGREE ${ cn_degree }
#define VN_DEGREE ${ vn_degree }
#define LLR_MAX 150

double boxplus(double a,double b)
{
double boxp=log((1+exp(a+b))/(exp(a)+exp(b)));
return sign(boxp)*min((double)LLR_MAX,(double) sign(boxp)*boxp);
}


void kernel send_channel_values_to_checknode_inbox(global const double* channel_values,
						   global const int* inbox_memory_start_varnodes,
						   global const int* degree_varnode_nr, 
						   global const int* target_memorycells_varnodes,
						   global double* checknode_inbox
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


void kernel checknode_update(global const double* checknode_inbox,
				   global const int* inbox_memory_start_checknodes,
				   global const int* degree_checknode_nr,
				   global const int* target_memorycells_checknodes,
				   global double* varnode_inbox)

{
  int gid1=get_global_id(0);
  int gid2=get_global_id(1);

  int lid=get_local_id(0);
  int local_size=get_local_size(0);

  int node_degree=degree_checknode_nr[gid1];

  private double current_msgs[CN_DEGREE-1];


  int msgs_start_idx=inbox_memory_start_checknodes[gid1];

  for (int w=0;w!=node_degree;w++)
  {
  int target_memcell=target_memorycells_checknodes[inbox_memory_start_checknodes[gid1]+w];

  int pos=0;
  for(int v=0;v!=node_degree;v++)
  {
    if(v!=w) current_msgs[pos++]=checknode_inbox[(msgs_start_idx+v)*${ msg_at_time }+gid2];
  }


  double t_lm1=current_msgs[0];
  for(int l=0;l<node_degree-2;l++)
  {
    t_lm1=boxplus(current_msgs[l+1],t_lm1);

  }
  varnode_inbox[target_memcell*${ msg_at_time }+gid2]=sign(t_lm1)*min((double) LLR_MAX, (double)sign(t_lm1)*t_lm1);
  }
}




void kernel varnode_update(global const double* channel_values,
			   global const double* varnode_inbox,
			   global const int* inbox_memory_start_varnodes,
		       global const int* degree_varnode_nr,
			   global const int* target_memorycells_varnodes,
		       global double* checknode_inbox )
{
  int gid1=get_global_id(0);
  int gid2=get_global_id(1);

  int lid=get_local_id(0);
  int local_size=get_local_size(0);
  
  int node_degree=degree_varnode_nr[gid1];
  
  private double current_msgs[VN_DEGREE];
  
  int msgs_start_idx=inbox_memory_start_varnodes[gid1];

  current_msgs[0]=channel_values[gid1*${ msg_at_time }+gid2];

  for (int w=0;w!=node_degree;w++)
  {
  int target_memcell=target_memorycells_varnodes[inbox_memory_start_varnodes[gid1]+w];

  int pos=1;
  for(int v=0;v!=node_degree;v++)
  {
    if(v!=w){
      //printf("pos %d %d\n",pos, node_degree);
      current_msgs[pos]=varnode_inbox[(msgs_start_idx+v)*${ msg_at_time }+gid2];
      pos=pos+1;

    }  
  }


  double t_0=current_msgs[0]+current_msgs[1];
  double t_lm1=t_0;
  for(int l=1;l<=node_degree-2;l++)
  {
    t_lm1=t_lm1+current_msgs[l+1];
  }

  checknode_inbox[target_memcell*${ msg_at_time }+gid2]=sign(t_lm1)*min((double) LLR_MAX, (double)sign(t_lm1)*t_lm1);
  }
  
}


void kernel checknode_update_minsum(global const double* checknode_inbox,
				   global const int* inbox_memory_start_checknodes,
				   global const int* degree_checknode_nr, 
				   global const int* target_memorycells_checknodes,
				   global double* varnode_inbox)
{
  int gid1=get_global_id(0);
  int gid2=get_global_id(1);

  int lid=get_local_id(0);
  int local_size=get_local_size(0);
  
  int node_degree=degree_checknode_nr[gid1];
  
  private double current_msgs[CN_DEGREE-1];
  
  
  int msgs_start_idx=inbox_memory_start_checknodes[gid1];
    
  for (int w=0;w!=node_degree;w++)
  {
  int target_memcell=target_memorycells_checknodes[inbox_memory_start_checknodes[gid1]+w];
  
  int pos=0;
  for(int v=0;v!=node_degree;v++)
  {
    if(v!=w) current_msgs[pos++]=checknode_inbox[(msgs_start_idx+v)*${ msg_at_time }+gid2];
  }
  
  
  double t_lm1=current_msgs[0];
  for(int l=0;l<node_degree-2;l++)
  {
    t_lm1=sign(current_msgs[l+1]*t_lm1)*min(sign(t_lm1)*t_lm1,sign(current_msgs[l+1])*current_msgs[l+1]);

  }



  varnode_inbox[target_memcell*${ msg_at_time }+gid2]=t_lm1;
  }
}


void kernel calc_varnode_output(global const double* channel_values,
			   global const double* varnode_inbox,
			   global const int* inbox_memory_start_varnodes,
		       global const int* degree_varnode_nr,
		       global double* varnode_output)
{
  int gid=get_global_id(0);
  int gid1=get_global_id(0);
  int gid2=get_global_id(1);

  int lid=get_local_id(0);
  int local_size=get_local_size(0);

  int node_degree=degree_varnode_nr[gid1];
  
  private double current_msgs[VN_DEGREE+1];
  
  int msgs_start_idx=inbox_memory_start_varnodes[gid1];
  current_msgs[0]=channel_values[gid1*${ msg_at_time }+gid2];
  

  for(int v=0;v!=node_degree;v++)
  {
    current_msgs[v+1]=varnode_inbox[(msgs_start_idx+v)*${ msg_at_time }+gid2];
  }

  double t_0=current_msgs[0]+current_msgs[1];
  double t_lm1=t_0;
  for(int l=1;l!=node_degree;l++)
  {
    t_lm1=t_lm1+current_msgs[l+1];
  }
  varnode_output[gid1*${ msg_at_time }+gid2]=t_lm1;
  
}

void kernel calc_syndrome(global const double* checknode_inbox,
			  global const int* inbox_memory_start_checknodes,
			  global const int* degree_checknode_nr,			  
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
    mod_sum+=(int)( checknode_inbox[(start_idx+w)*${ msg_at_time }+gid2 ]<0);
    mod_sum=mod_sum%2;
  }
  syndrome[gid1*${ msg_at_time }+gid2]=mod_sum;
}