#include <iostream>
#include <cstring>
#include <cmath>
#include <stdio.h>
#include <omp.h>
#include <sys/time.h>
# include <mpi.h>

//namespace
using namespace std;
//constants and other global variables
const double e_temp=300; //eviroment temperature
const double k=1.3806505E-23;//Boltzmann constant
const double lc=4.05E-10;//lattice constant
const double mol=6.02e23;// atoms per mol
const double ram=26.98;// relative atomic mass
const double m=ram/mol/1000; // mass of a atom(kg)
const double dens=2.7E3; //density (kg/m^3)
const double PI=3.1415927;
const double r_cut=4*lc;
const double dt=1E-15;// delta time(s)
const double tr=4E-13;
//constants for morse
const double De=4.32264E-20;
const double alpha=1.0341E10;
const double r0=3.4068E-10;
int wcount=50;
double redge=wcount*lc;
double tedge;
//functions
void init(string type,double atoms[], double speeds[], size_t N);
void maxwell(double maxwells[], double mw_v[] ,int N);
void edge(double atoms[],double v[],int N);
void get_acc(double atoms[],double acc[], int N);
void laser(double atoms[],double v[],int N);
template<typename T> T myabs(T x);
//main
int main(int argc,char* argv[]){
	size_t N=5000;
	int iter1=100;
	int iter2=100;
	size_t SIZE=N*2;
	string AtomType="AL";//for extension in the future, for now it is useless
	double atoms[SIZE];
	double v[SIZE];
	init(AtomType,atoms, v, SIZE);
	N=N*2;
	double acc[N];
	for(int i=0;i<N;i++){
		acc[i]=0;
	}

	double T_trans[N];
	double B_trans[N];
	int ttrans_index=0;
	int btrans_index=0;
	//calculate the speed and coordination based on Leap_Frog algorithm
	get_acc(atoms,acc,N);
	for(int i=0;i<N;i++){
		v[i]-=acc[0]*0.5*dt;
	}

	struct timeval start_time, stop_time, elapsed_time;  // timers
	gettimeofday(&start_time,NULL); // Unix timer

	for(int it=0;it<iter1;it++){
		edge(atoms,v,N);
		//cout<< atoms[N-4] << "  " << atoms[N-3] << "  ";
		int i;
		#pragma omp parallel for
		for(i=0;i<N;i++){//seems less accurate to me, but articles say that it's more accurate.
			//if(i/2%(2*wcount)!=0 && i/2%(2*wcount)!=2*wcount-1){
			v[i]+=acc[i]*dt;
			atoms[i]+=v[i]*dt;
			acc[i]=0;
			//}
		}
		get_acc(atoms,acc,N);
	}
	gettimeofday(&stop_time,NULL);
	timersub(&stop_time, &start_time, &elapsed_time); // Unix time subtract routine
	cout << elapsed_time.tv_sec+elapsed_time.tv_usec/1000000.0 << "q  ";
	//laser(atoms,v,N);
	int ID, received, Nprocess;
	MPI_Status status;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &ID);
	MPI_Comm_size(MPI_COMM_WORLD, &Nprocess);
	double T_s=tedge/Nprocess*(ID+1);
	double B_s=tedge/Nprocess*ID;
	if(ID==0){
		B_s-=tedge*3;
	}
	else if (ID==Nprocess-1){
		T_s+=tedge*3;
	}
	int Start=(ID*N/2)/Nprocess;
	int Stop=((ID+1)*N/2)/Nprocess;
	int N_Job=(Stop-Start)*2;
	for(int i=0;i<N_Job;i++){
		atoms[i]=atoms[i+Start];
		v[i]=v[i+Start];
		acc[i]=acc[i+Start];
	}

		gettimeofday(&start_time,NULL); // Unix timer
	for(int it=0;it<iter2;it++){

			double atom_temp{};
			ttrans_index=0;
			btrans_index=0;
			int s_index=0;
	for(int i=0;i<N_Job;i+=2){
		if(atoms[s_index+1]>T_s && ID!=Nprocess-1){
			T_trans[ttrans_index]=atoms[s_index];
			T_trans[++ttrans_index]=atoms[s_index+1];
			T_trans[++ttrans_index]=v[s_index];
			T_trans[++ttrans_index]=v[s_index+1];
			s_index+=2;
			N_Job-=2;
			i-=2;
			
		}
		else if(atoms[s_index+1]<B_s && ID != Nprocess-1){
			B_trans[btrans_index]=atoms[s_index];
			B_trans[++btrans_index]=atoms[s_index+1];
			B_trans[++btrans_index]=v[s_index];
			B_trans[++btrans_index]=v[s_index+1];
			s_index+=2;
			N_Job-=2;
			i-=2;
		}	
		atoms[i+2]=atoms[s_index];
		atoms[i+3]=atoms[s_index+1];
		v[i+2]=v[s_index];
		v[i+3]=v[s_index+1];
		s_index+=2;
	
	}

	#pragma omp parallel for
	for(int i=0;i<N_Job;i+=2){
		//periodic boundary
		#pragma omp atomic
		atom_temp+=pow(v[i],2)+pow(v[i],2);

		if(atoms[i]>redge)
			atoms[i]-=redge;
		if(atoms[i]<0)
			atoms[i]+=redge;
		
	}
	
	atom_temp=atom_temp*m/k/N_Job*1.5;
	//Then Berendsen heat bath
	double C_hb=sqrt(1+dt/tr*(e_temp/atom_temp-1));
	#pragma omp parallel for
	for(int i=0;i<N_Job;i++){
		v[i]*=C_hb;
	}
	//cout <<atom_temp<< endl;
	
	if(ID != Nprocess-1){
		++ttrans_index;
		MPI_Send(&ttrans_index,1, MPI_INT, ID+1, 1, MPI_COMM_WORLD);
		if(ttrans_index>1)
			MPI_Send(&T_trans[0],ttrans_index, MPI_DOUBLE, ID+1, 2, MPI_COMM_WORLD);
		MPI_Recv(&btrans_index,1, MPI_INT,ID+1,3, MPI_COMM_WORLD, &status);
		if(btrans_index>1){
			double dis[btrans_index];
			MPI_Recv(&dis[0],btrans_index, MPI_DOUBLE,ID+1,4, MPI_COMM_WORLD, &status);
			for(int i=0; i<btrans_index;i+=4){
				atoms[i+N_Job]=dis[i];
				atoms[i+N_Job+1]=dis[i+1];
				v[i+N_Job]=dis[i+2];
				v[i+N_Job+1]=dis[i+3];
			}
			N_Job+=btrans_index/2;
		}

	}
	if(ID != 0){
		++btrans_index;
		MPI_Send(&btrans_index,1, MPI_INT, ID-1, 3, MPI_COMM_WORLD);
		if(btrans_index>1)
			MPI_Send(&T_trans[0],btrans_index, MPI_DOUBLE, ID-1, 4, MPI_COMM_WORLD);
		MPI_Recv(&ttrans_index,1, MPI_INT,ID-1,1, MPI_COMM_WORLD, &status);
		if(ttrans_index>1){
			double dis[btrans_index];
			MPI_Recv(&dis[0],ttrans_index, MPI_DOUBLE,ID-1,2, MPI_COMM_WORLD, &status);
			for(int i=0; i<btrans_index;i+=4){
				atoms[i+N_Job]=dis[i];
				atoms[i+N_Job+1]=dis[i+1];
				v[i+N_Job]=dis[i+2];
				v[i+N_Job+1]=dis[i+3];
			}
			N_Job+=ttrans_index/2;
		}
	}
		//cout<< atoms[N-4] << "  " << atoms[N-3] << "  ";
		int i;
		#pragma omp parallel for
		for(i=0;i<N_Job;i++){//seems less accurate to me, but articles say that it's more accurate.
			//if(i/2%(2*wcount)!=0 && i/2%(2*wcount)!=2*wcount-1){
			v[i]+=acc[i]*dt;
			atoms[i]+=v[i]*dt;
			acc[i]=0;
			//}
		}
		get_acc(atoms,acc,N_Job);

	}
	MPI_Finalize();
	gettimeofday(&stop_time,NULL);
	timersub(&stop_time, &start_time, &elapsed_time); // Unix time subtract routine
	cout << elapsed_time.tv_sec+elapsed_time.tv_usec/1000000.0 << "  \n";
	//swap acc and acc_t
	//cout <<acc[1] <<endl;
	//cout <<acc_t[1] <<endl;
	
	return 0;
}
//functions
void init(string type,double atoms[], double speeds[], size_t N){//initialize the model
	double types[]{0,0,0.5,0.5};
	size_t size_types=sizeof(types)/sizeof(double);
	int height{};
	if(wcount%2!=0)
		--wcount;
	for(int i=0;i<N;i=i+size_types){//initialize position
		for(int j=0;j<size_types;j++){
			if(j%2==0)
			atoms[i+j]=(i/size_types%wcount+types[j])*lc;
			else
			atoms[i+j]=(types[j]+height)*lc;
		}
		if(i/size_types%wcount==wcount-1){
			++height;
		}
	}
	tedge=height*lc;
	int Nmaxwell=1000;
	int rc=100000;// random control(should much bigger than Nmaxwell)
	double rt=rc;// random control(should much bigger than Nmaxwell)
	double maxwells[Nmaxwell]; //use for converting rand() to maxwell distribustion
	double mw_velocity[Nmaxwell];//just for reduce the calculation
	maxwell(maxwells,mw_velocity,Nmaxwell);
	for(int i=0;i<N;i++){//convert rand() to maxwell distribustion and initialize speed
		double x=(rand()%(2*rc+1)-rt)/rt;
		double xx= myabs(x);
		int j{};
		while(maxwells[j]<xx)
			j++;
		speeds[i]=x>0 ? mw_velocity[j-1]: -mw_velocity[j-1];
	}
}

void maxwell(double maxwells[], double mw_v[] ,int N){ //Maxwell distribution
	srand((int)time(0));
	const double ac=100;//accuracy control
	const double c1=sqrt(m/(4/3*PI*k*e_temp));
	const double c2=-m/(4/3*k*e_temp);
	const double max_v=sqrt(k*e_temp/m*2/3)*3;//with enough coverage
	const double dv=max_v/(N*ac);
	double f;
	double f1=c1;
	double f2;
	double v{};
	for(int i=0; i<N; i++){
		for(int j=0; j<ac;j++){//accuracy control
			f2=c1*exp(c2*pow(v,2));
			f+=(f1+f2)*dv;//doubled as there are two direction
			v+=dv;
			f1=f2;
		}
		maxwells[i]=f;
		mw_v[i]=v;
	}
	for(int i=0;i<N;i++){//make it from 0 to 1
		maxwells[i]/=f;
	}
}

void get_acc(double atoms[],double acc[], int N){	//calculate the acceleration by morse potential
		//#pragma omp parallel for //collapse (1)  // not good enough
		#pragma omp parallel//faster
		{
			int nstart=omp_get_thread_num()*2;
			int step=omp_get_num_threads()*2;
			double acc_all,acc_x,acc_y,d_x,d_y,distance;
			for(int i=nstart;i<N;i+=step){
			for(int j=i+2;j<N;j+=2){
				//reduce calculation,way faster
				//with 25*25*2 atoms, about 10 times faster with almost the same result
				d_y=atoms[i+1]-atoms[j+1];
				if(d_y > r_cut ||d_y < -r_cut)
					continue;
				d_x=atoms[i]-atoms[j];
				d_x=myabs(d_x) < redge-myabs(d_x) ? d_x : (d_x > 0 ? d_x-redge : d_x+redge);//periodic

				if(d_x > r_cut ||d_x < -r_cut)
					continue;
				distance=sqrt(pow(d_x,2)+pow(d_y,2));
				if(distance > r_cut)
					continue;
				acc_all=2*De*alpha*exp(-alpha*(distance-r0))*(1-exp(-alpha*(distance-r0)))/m;
				acc_x=acc_all*d_x/distance;
				acc_y=acc_all*d_y/distance;
				#pragma omp atomic
				acc[i]-=acc_x;
				#pragma omp atomic
				acc[j]+=acc_x;
				#pragma omp atomic
				acc[i+1]-=acc_y;
				#pragma omp atomic
				acc[j+1]+=acc_y;
			}
		}
	}
}

template<typename T> T myabs(T x){
	return x>0 ? x : -x;
}

void laser(double atoms[],double v[],int N){//from the source code, transfer laser energy to atoms, there are a few changes
	const double AA = 0.1; //穿透率
	const double alpha = 0.12E9; //材料吸收系数
	const double w0 = 12.4E-6;//束腰半径
	const double F = 100; // 单位 J/m^2 光强
	const double lasert = 1E-13; //激光作用时�?
	const double ww = w0*w0;
	const double V =m/dens; //volume
	const double Im = AA*alpha*sqrt(4*log(2)/PI)*F;
	#pragma omp parallel for collapse (1)
	for (int i = 0; i < N; i+=2) {
		double r = redge / 2 - atoms[i];//到激光中心的距离
		for (double t = 0; t < lasert; t += dt) {
			double deltaE = Im*exp(-alpha * (tedge-atoms[i+1]) - 4 * log(2)*pow((t - lasert) / t, 2) - r*r / ww)*V;//能量增加�?
			double E = 0.5*m*(pow(v[i], 2) + pow(v[i+1], 2)); //?��本的能�?
			double rate = sqrt((E + deltaE) / E);//速度变化比例
			v[i] *= rate;//激光作用后的速度
			v[i+1] *= rate;
		}
	}
}

void edge(double atoms[],double v[],int N){//position control and temperature control
	double atom_temp{};
#pragma omp parallel for
	for(int i=0;i<N;i+=2){

		//periodic boundary
		#pragma omp atomic
		atom_temp+=pow(v[i],2)+pow(v[i+1],2);

		if(atoms[i]>redge)
			atoms[i]-=redge;
		if(atoms[i]<0)
			atoms[i]+=redge;
	}
	atom_temp=atom_temp*m/k/N*1.5;
	//Then Berendsen heat bath
	double C_hb=sqrt(1+dt/tr*(e_temp/atom_temp-1));
	#pragma omp parallel for
	for(int i=0;i<N;i++){
		v[i]*=C_hb;
	}
	
//	cout <<atom_temp<< endl;
}

