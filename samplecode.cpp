/* All rights reserved by Ligesh Theeyancheri & Jennifer Schwarz*/
#include <sys/stat.h>
#include <math.h>
#include <algorithm>
#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <vector>
#include <ctime>
#include <cmath>
#include <random>
#include <complex>
#include <iomanip>
#include <map> 

using namespace std;

// control input file
int INITIAL_FILE=101;

//control output file
int SAMPLE_NUMBER=0;// REMEMBER TO CHANGE EVERYTIME WHEN USED!!!!!!!!!!!!
// Next label: 110;

// system control
const double Diffusion=1;
const double F_thermal=1;
const double timestep=0.0001; //time step
double total_time=1000; //total time
double running_time=0.0;

// parameters of Monomers
const int SYSTEMSIZE=10000; //Number of CRMT monomers
const double a_CRMT=1.0; //restlength between monomers along CRMT chain
const double r_CRMT=0.51299; //radius of CRMT monomer
const double k_CRMT=140.0; //spring constant between CRMT monomers, also soft repulsion stiffness

// parameters of Lamina shell
const bool SHELL_IS_SOFT=true;
const int SYSTEMSIZE2=6000; //Number of shell monomers
const double MapCoeff=10.0;//4.7 for 512 monomers; // shell monomers stored are on a unit sphere so they need to be mapped to a larger sphere surface
const double r_shell=r_CRMT; //when there are 500/1000/5000 shell monomers, distance between shell monomers is around 0.138/0.0978/0.05
const double k_shell=k_CRMT; // spring constant between shell monomers

// parameters of Cell shell
const bool SHELL_IS_SOFT=true;
const int SYSTEMSIZE3=6000;
const double MapCoeffOuter=10.0;
const double r_outershell=r_CRMT;
const double k_outershell=k_CRMT;
 
// parameters of chromatin crosslinks
const int List_Num_xlink[6]={0,100,500,1000,2000,2500};
const double k_crosslink=k_CRMT;
int Num_crosslink;
double slope=1.0; // slope of the linear crosslinking

// parameters of motors
bool MOTOR_IS_ON;
const bool MOTOR_IS_FIXED=false;
const double R_0_CRMT=1.5*r_CRMT; //Motor's radius
const double List_F0[5]={0,5,10,30,100};
double F_0_CRMT; // Motor's maximum force
const int Num_motor=500; //total number of motors
const int Num_Con=0; // number of contractile motors
int Num_Ext=Num_motor-Num_Con; // number of extensile motors
const double Thresh=2*R_0_CRMT; // distance<Thresh between CRMT monomers: defined as neighbors, stored in neighbor lists
const double rate_off=0.001; //0.005;
const int UpdateMotorCycle=1000; //After # updates, update motors 

// LINK between shell and Monomer
const double k_D_S=k_CRMT; // spring constant between shell monomers and CRMT monomers
const double List_specialtwo[2][4]={{0,r_CRMT+r_shell-0.035,r_CRMT+r_shell,r_CRMT+r_shell+0.08},{0,r_CRMT+r_shell-0.035,r_CRMT+r_shell+0.05,r_CRMT+r_shell+0.25}};
const double List_Thresh_D_S[5]={0,0.4,0.48,0.57,0.65};
double Thresh_D_S; // distance<this between CRMT monomer and shell monomer: link together
const double Lambda=r_shell+r_CRMT+20.0; // distance>this between CRMT monomer and shell monomer: break


//LINk between Nuclear Lamina and Cell Shell
const double k_C_N=k_CRMT;
const double shells_cutoff = 2.0;
int N_Cell_Lamin;
const int List_Num_Shelllinks[8]={0,500,1000,1500,2000,2500,4000,4500};
const int List_BeadsperChain[5]={1, 2, 3, 4, 5} 

// parameters of spherical shell
const double K_Inner=5; //Inner: spherical potential to keep shell stiff(100 or 0)
const double R_Inner=MapCoeff*1.0;

// define basic monomer structure
struct Particles
{
    int node;
    double r;
    double x;
    double y;
    double z;
    double fx;
    double fy;
    double fz;
    double k;
    int status;
    int hashvalue;
    struct Particles *next;
    std::vector<int> springs;
    std::vector<double> k_s;
    std::vector<double> a_s;
};

// define polymer, shells and hash list
struct Particles Monomer[SYSTEMSIZE+SYSTEMSIZE2++SYSTEMSIZE3];
const double Mesh_size=2*r_CRMT;
const double MaxMesh=60;
const int Mesh_num=int(MaxMesh/Mesh_size);
const int MaxHashSize=int(Mesh_num*Mesh_num*Mesh_num);
struct Particles *HashchartHead[MaxHashSize]={nullptr};

// spherical boundary condition
const bool BOUNDARY1_IS_ON=false;
const double K_boundary1=k_CRMT; // spherical boundary stiffness
const double R_Boundary1_begin=10;//9.55184;//6.37808; //for 2/3 strain //4.46876; // for 512
const double R_Boundary1_end=MapCoeff;//4.46876;
const double v_B1=0;
double R_Boundary1=R_Boundary1_begin;
double v_Boundary1=-v_B1;

// flat boundary condition
const bool BOUNDARY2_IS_ON=false;
const double K_boundary2=5.0; // flat boundary stiffness
const double R_Boundary2_begin=5;
const double R_Boundary2_end=1.48959;
const double v_B2=0.05;
double R_Boundary2=R_Boundary2_begin;
double v_Boundary2=-v_B2;

// output control
const int Plot_cycle=10000;
const int Plot_cycle1=5000;
double Plot_label=0;

// random seed set up
std::default_random_engine generator_seed(time(NULL));
std::uniform_int_distribution<long long> distribution_uni_seed(0,5000000);

void clearF(int Ni, int Nf){
    int n;
    for(n=Ni;n<Nf;n++){
        Monomer[n].fx=0;
        Monomer[n].fy=0;
        Monomer[n].fz=0;
    }
}

void Polymer_spring(){
    int n;
    double distance,exceed;
    for(n=0;n<SYSTEMSIZE-1;n++){
        distance=sqrt(pow(Monomer[n].x-Monomer[n+1].x,2)+pow(Monomer[n].y-Monomer[n+1].y,2)+pow(Monomer[n].z-Monomer[n+1].z,2));
        exceed=distance-1.0;
        Monomer[n].fx+=exceed*k_CRMT*(Monomer[n+1].x-Monomer[n].x)/distance;
        Monomer[n].fy+=exceed*k_CRMT*(Monomer[n+1].y-Monomer[n].y)/distance;
        Monomer[n].fz+=exceed*k_CRMT*(Monomer[n+1].z-Monomer[n].z)/distance;
        Monomer[n+1].fx-=exceed*k_CRMT*(Monomer[n+1].x-Monomer[n].x)/distance;
        Monomer[n+1].fy-=exceed*k_CRMT*(Monomer[n+1].y-Monomer[n].y)/distance;
        Monomer[n+1].fz-=exceed*k_CRMT*(Monomer[n+1].z-Monomer[n].z)/distance;
    }
}

void Other_spring(int Ni, int Nf){
    int n,n2,k;
    double distance,exceed;
    for(n=Ni;n<Nf;n++){
        for(k=0;k<Monomer[n].springs.size();k++){
            n2=Monomer[n].springs[k];
            distance=sqrt(pow(Monomer[n].x-Monomer[n2].x,2)+pow(Monomer[n].y-Monomer[n2].y,2)+pow(Monomer[n].z-Monomer[n2].z,2));
            exceed=distance-Monomer[n].a_s[k];
            Monomer[n].fx+=exceed*Monomer[n].k_s[k]*(Monomer[n2].x-Monomer[n].x)/distance;
            Monomer[n].fy+=exceed*Monomer[n].k_s[k]*(Monomer[n2].y-Monomer[n].y)/distance;
            Monomer[n].fz+=exceed*Monomer[n].k_s[k]*(Monomer[n2].z-Monomer[n].z)/distance;
        }
    }
}

void Spherical_Constraint(int Ni, int Nf){
    int n;
    double distance,exceed;
    for(n=Ni;n<Nf;n++){
        distance=sqrt(pow(Monomer[n].x,2)+pow(Monomer[n].y,2)+pow(Monomer[n].z,2));
        exceed=distance-R_Inner;
        Monomer[n].fx-=exceed*K_Inner*Monomer[n].x/distance;
        Monomer[n].fy-=exceed*K_Inner*Monomer[n].y/distance;
        Monomer[n].fz-=exceed*K_Inner*Monomer[n].z/distance;
    }
}

void Boundary1_Constraints(int Ni, int Nf){
    int n;
    double distance,exceed;
    for(n=Ni;n<Nf;n++){
        distance = sqrt(pow(Monomer[n].x,2)+pow(Monomer[n].y,2)+pow(Monomer[n].z,2));
        exceed=distance+Monomer[n].r-R_Boundary1;
        if (exceed>0){
            Monomer[n].fx-=exceed*K_boundary1*Monomer[n].x/distance;
            Monomer[n].fy-=exceed*K_boundary1*Monomer[n].y/distance;
            Monomer[n].fz-=exceed*K_boundary1*Monomer[n].z/distance;
        }
    }
}

void ThermalFluctuation(int Add_on_random_time, int Ni, int Nf){ // random and indepedent in xyz directions
    int n;
    std::default_random_engine generator(time(NULL)+Add_on_random_time+distribution_uni_seed(generator_seed));
    std::normal_distribution<double> distribution(0.0,1.0);
    if(running_time==0){
        char filename_Monomer[50];
        sprintf (filename_Monomer, "test_rad.txt");
        std::ofstream outfile_Monomer(filename_Monomer);
        for(n=0;n<100000;n++){
            outfile_Monomer << distribution(generator) << std::endl;
        }
        outfile_Monomer.close();
    }
    for(n=Ni;n<Nf;n++){
        Monomer[n].fx+=(sqrt(2/Diffusion/timestep)*F_thermal*distribution(generator));
        Monomer[n].fy+=(sqrt(2/Diffusion/timestep)*F_thermal*distribution(generator));
        Monomer[n].fz+=(sqrt(2/Diffusion/timestep)*F_thermal*distribution(generator));
    }
}

// together with function below, to solve forces due to soft repulsion between monomers
void ComputeRepulsion(int n, struct Particles *temp){
    double distance;
    while(temp!=nullptr){
        if(Monomer[n].hashvalue!=temp->hashvalue || n>temp->node){
            distance=sqrt(pow(Monomer[n].x-temp->x,2)+pow(Monomer[n].y-temp->y,2)+pow(Monomer[n].z-temp->z,2));
            if(distance<(Monomer[n].r+temp->r)){
                double overlap=Monomer[n].r+temp->r-distance;
                Monomer[n].fx+=(0.5*(Monomer[n].k+temp->k)*overlap*(Monomer[n].x-temp->x)/distance);
                Monomer[n].fy+=(0.5*(Monomer[n].k+temp->k)*overlap*(Monomer[n].y-temp->y)/distance);
                Monomer[n].fz+=(0.5*(Monomer[n].k+temp->k)*overlap*(Monomer[n].z-temp->z)/distance);
                temp->fx-=(0.5*(Monomer[n].k+temp->k)*overlap*(Monomer[n].x-temp->x)/distance);
                temp->fy-=(0.5*(Monomer[n].k+temp->k)*overlap*(Monomer[n].y-temp->y)/distance);
                temp->fz-=(0.5*(Monomer[n].k+temp->k)*overlap*(Monomer[n].z-temp->z)/distance);
            }
        }
        temp=temp->next;
    }
}

void UpdateSoftRepulsion(int Ni, int Nf){
    struct Particles *temp;
    int n;
    for(n=Ni;n<Nf;n++){
        temp=HashchartHead[Monomer[n].hashvalue];
        ComputeRepulsion(n,temp);
        temp=HashchartHead[Monomer[n].hashvalue+1]; // x+1
        ComputeRepulsion(n,temp);
        temp=HashchartHead[Monomer[n].hashvalue+Mesh_num]; // y+1
        ComputeRepulsion(n,temp);
        temp=HashchartHead[Monomer[n].hashvalue+Mesh_num*Mesh_num]; // z+1
        ComputeRepulsion(n,temp);
        temp=HashchartHead[Monomer[n].hashvalue+Mesh_num+Mesh_num*Mesh_num]; // y+1,z+1
        ComputeRepulsion(n,temp);
        temp=HashchartHead[Monomer[n].hashvalue+1+Mesh_num*Mesh_num]; // x+1, z+1
        ComputeRepulsion(n,temp);
        temp=HashchartHead[Monomer[n].hashvalue+1+Mesh_num]; // x+1, y+1
        ComputeRepulsion(n,temp);
        temp=HashchartHead[Monomer[n].hashvalue+1+Mesh_num+Mesh_num*Mesh_num]; // x+1,y+1,z+1
        ComputeRepulsion(n,temp);
    }
}

// together with function below, to solve forces due to active motors
void ComputeMotorForce(int n, struct Particles *temp){
    double distance;
    while(temp!=nullptr){
        if(temp->node<SYSTEMSIZE){
            if(Monomer[n].status!=0 || temp->status!=0){
                if(Monomer[n].hashvalue!=temp->hashvalue || n>temp->node){
                    distance=sqrt(pow(Monomer[n].x-temp->x,2)+pow(Monomer[n].y-temp->y,2)+pow(Monomer[n].z-temp->z,2));
                    if(distance<Thresh){
                        if(Monomer[n].status!=0){
                            temp->fx+=(Monomer[n].status*F_0_CRMT*(temp->x-Monomer[n].x)/distance);
                            temp->fy+=(Monomer[n].status*F_0_CRMT*(temp->y-Monomer[n].y)/distance);
                            temp->fz+=(Monomer[n].status*F_0_CRMT*(temp->z-Monomer[n].z)/distance);
//                            Monomer[n].fx-=(Monomer[n].status*F_0_CRMT*(temp->x-Monomer[n].x)/distance);
//                            Monomer[n].fy-=(Monomer[n].status*F_0_CRMT*(temp->y-Monomer[n].y)/distance);
//                            Monomer[n].fz-=(Monomer[n].status*F_0_CRMT*(temp->z-Monomer[n].z)/distance);
                        }
                        if(temp->status!=0){
                            Monomer[n].fx-=(temp->status*F_0_CRMT*(temp->x-Monomer[n].x)/distance);
                            Monomer[n].fy-=(temp->status*F_0_CRMT*(temp->y-Monomer[n].y)/distance);
                            Monomer[n].fz-=(temp->status*F_0_CRMT*(temp->z-Monomer[n].z)/distance);
//                            temp->fx+=(temp->status*F_0_CRMT*(temp->x-Monomer[n].x)/distance);
//                            temp->fy+=(temp->status*F_0_CRMT*(temp->y-Monomer[n].y)/distance);
//                            temp->fz+=(temp->status*F_0_CRMT*(temp->z-Monomer[n].z)/distance);
                        }
                    }
                }
            }
        }
        temp=temp->next;
    }
}
void UpdateMotorForce(int Ni, int Nf){
    struct Particles *temp;
    int n,i,j,k;
    for(n=Ni;n<Nf;n++){
        for(i=0;i<3;i++){
            for(j=0;j<3;j++){
                for(k=0;k<3;k++){
                    temp=HashchartHead[Monomer[n].hashvalue+1*i+Mesh_num*j+Mesh_num*Mesh_num*k];
                    ComputeMotorForce(n,temp);
                }
            }
        }
    }
}

void Updatexyz(int Ni, int Nf){
    int n;
    for(n=Ni;n<Nf;n++){
        Monomer[n].x+=(Monomer[n].fx*timestep*Diffusion);
        Monomer[n].y+=(Monomer[n].fy*timestep*Diffusion);
        Monomer[n].z+=(Monomer[n].fz*timestep*Diffusion);
    }
}

void UpdateHash(int Ni, int Nf){ // must call this function after updating x,y,z
    int n;
    for(n=Ni;n<Nf;n++){
        int newhashvalue=int((Monomer[n].x+0.5*MaxMesh)/Mesh_size)+Mesh_num*int((Monomer[n].y+0.5*MaxMesh)/Mesh_size)+Mesh_num*Mesh_num*int((Monomer[n].z+0.5*MaxMesh)/Mesh_size);
        if(newhashvalue!=Monomer[n].hashvalue){
            int temp=(HashchartHead[Monomer[n].hashvalue]->node);
            // delete from old hash chain
            if(temp==n){
                HashchartHead[Monomer[n].hashvalue]=Monomer[n].next;
            }
            else{
                bool Found_n=false;
                while(!Found_n){
                    if(Monomer[temp].next->node==n){
                        Found_n=true;
                    }
                    else{temp=Monomer[temp].next->node;}
                }
                Monomer[temp].next=Monomer[n].next;
            }
            // add to the first in new hash chain
            Monomer[n].next=HashchartHead[newhashvalue];
            HashchartHead[newhashvalue]=&(Monomer[n]);
            Monomer[n].hashvalue=newhashvalue;
        }
    }
}

void Updatemotor(int Add_on_random_time){
    std::default_random_engine generator(time(NULL)+Add_on_random_time+distribution_uni_seed(generator_seed));
    std::uniform_int_distribution<long long> distribution_uni(0,50000000);
    
    int i,j,head,tail,ind;
    int randlist[SYSTEMSIZE];
    int go[SYSTEMSIZE];
    std::vector<int> Rest;
    
    head=0;
    tail=SYSTEMSIZE;
    for(i=0;i<SYSTEMSIZE;i++){
        if(Monomer[i].status!=0){
            if(int(distribution_uni(generator)%1000)>=int(rate_off*1000)){
                go[i]=head;
                head++;
            }
            else{
                go[i]=tail-1;
                tail--;
            }
        }
        else{Rest.push_back(i);go[i]=-1;}
    }
    while(Rest.size()>1){ // randomly pick out element until there is only one left
        ind=int(distribution_uni(generator)%(Rest.size()));
        go[Rest[ind]]=head;
        head++;
        Rest.erase(Rest.begin()+ind);
    }
    go[Rest[0]]=head; // the last one
    
    for(i=0;i<SYSTEMSIZE;i++){randlist[go[i]]=i;}
    head=tail+Num_motor-SYSTEMSIZE;
    
    for(i=tail+Num_motor-SYSTEMSIZE;i<Num_motor;i++){
        Monomer[randlist[i]].status=Monomer[randlist[i+SYSTEMSIZE-Num_motor]].status;
        Monomer[randlist[i+SYSTEMSIZE-Num_motor]].status=0;
    }
}

 // Function to calculate the center of the shell
std::vector<double> calculateShellCenter() {
    std::vector<double> center(3, 0.0);
    for (int i = SYSTEMSIZE; i < SYSTEMSIZE + SYSTEMSIZE2; i++) {
        center[0] += Monomer[i].x;
        center[1] += Monomer[i].y;
        center[2] += Monomer[i].z;
    }
    center[0] /= SYSTEMSIZE2;
    center[1] /= SYSTEMSIZE2;
    center[2] /= SYSTEMSIZE2;
    return center;
}

void createCrosslinks(int Num_crosslink, double Thresh) {
    std::vector<std::vector<int>> Neighborlist(SYSTEMSIZE);
    std::vector<double> center = calculateShellCenter();
    std::vector<std::pair<int, double>> particles_with_distances;
    double max_distance = 0;

    // First pass: calculate distances and build neighbor list
    for (int i = 0; i < SYSTEMSIZE; i++) {
        double dx = Monomer[i].x - center[0];
        double dy = Monomer[i].y - center[1];
        double dz = Monomer[i].z - center[2];
        double dist_from_center = std::sqrt(dx*dx + dy*dy + dz*dz);
        
        particles_with_distances.push_back({i, dist_from_center});
        max_distance = std::max(max_distance, dist_from_center);
        
        for (int j = i + 1; j < SYSTEMSIZE; j++) {
            double dist = std::sqrt(std::pow(Monomer[i].x-Monomer[j].x,2) +
                                    std::pow(Monomer[i].y-Monomer[j].y,2) +
                                    std::pow(Monomer[i].z-Monomer[j].z,2));
            if (dist < Thresh) {
                Neighborlist[i].push_back(j);
                Neighborlist[j].push_back(i);
            }
        }
    }

    // Sort particles by distance from center
    std::sort(particles_with_distances.begin(), particles_with_distances.end(),
              [](const auto& a, const auto& b) { return a.second < b.second; });

    const int NUM_SHELLS = 40; // Adjust as needed
    std::vector<int> crosslinks_per_shell(NUM_SHELLS, 0);
    double shell_width = max_distance / NUM_SHELLS;

    std::vector<double> shell_densities(NUM_SHELLS);
    double max_density = 0;

    // First pass: calculate unnormalized densities
    for (int i = 0; i < NUM_SHELLS; i++) {
        double shell_center = (i + 0.5) * shell_width;
        shell_densities[i] = slope * (shell_center / max_distance);
        max_density = std::max(max_density, shell_densities[i]);
    }

    // Second pass: normalize densities and calculate weights
    double total_weight = 0;
    std::vector<double> shell_weights(NUM_SHELLS);

    for (int i = 0; i < NUM_SHELLS; i++) {
        double outer_radius = (i + 1) * shell_width;
        double inner_radius = i * shell_width;
        double shell_volume = (4.0/3.0) * M_PI * (std::pow(outer_radius, 3) - std::pow(inner_radius, 3));
        
        // Normalize density
        double normalized_density = shell_densities[i] / max_density;

        shell_weights[i] = normalized_density * shell_volume;
        total_weight += shell_weights[i];
    }

    // Distribute crosslinks based on normalized weights
    for (int i = 0; i < NUM_SHELLS; i++) {
        crosslinks_per_shell[i] = std::round(Num_crosslink * shell_weights[i] / total_weight);
    }

    // Create crosslinks
    std::default_random_engine generator(time(NULL) + distribution_uni_seed(generator_seed));
    std::uniform_real_distribution<double> uniform_dist(0.0, 1.0);

    int current_shell = 0;
    int crosslinks_created = 0;
    std::map<double, int> crosslink_distribution;

    for (const auto& particle : particles_with_distances) {
        int i = particle.first;
        double dist_from_center = particle.second;

        while (dist_from_center > (current_shell + 1) * shell_width && current_shell < NUM_SHELLS - 1) {
            current_shell++;
        }

        if (crosslinks_created >= Num_crosslink) break;

        if (crosslinks_per_shell[current_shell] > 0 && !Neighborlist[i].empty()) {
            // Randomly select a neighbor
            int j_index = int(uniform_dist(generator) * Neighborlist[i].size());
            int j = Neighborlist[i][j_index];

            if (abs(i - j) < 2) continue;

            // Create crosslink
            double temp_dist = std::sqrt(std::pow(Monomer[i].x - Monomer[j].x, 2) +
                                         std::pow(Monomer[i].y - Monomer[j].y, 2) +
                                         std::pow(Monomer[i].z - Monomer[j].z, 2));

            Monomer[i].springs.push_back(j);
            Monomer[i].a_s.push_back(temp_dist);
            Monomer[i].k_s.push_back(k_crosslink);
            Monomer[j].springs.push_back(i);
            Monomer[j].a_s.push_back(temp_dist);
            Monomer[j].k_s.push_back(k_crosslink);

            crosslinks_created++;
            crosslinks_per_shell[current_shell]--;

            // Record the crosslink for distribution
            double rounded_distance = std::round(dist_from_center * 10.0) / 10.0;  // Round to 1 decimal place
            crosslink_distribution[rounded_distance]++;

            // Remove the used pair from neighbor lists to avoid duplicates
            Neighborlist[i].erase(Neighborlist[i].begin() + j_index);
            auto it_j = std::find(Neighborlist[j].begin(), Neighborlist[j].end(), i);
            if (it_j != Neighborlist[j].end()) {
                Neighborlist[j].erase(it_j);
            }
        }
    }

    // Save the crosslink distribution to a file
    char filename[400];
    sprintf(filename, "Soft_Cell_Nuclei_M%.2lf/INITIAL%dSAMPLE%dXLINK%dLAD%.2lfF%.1lfNM%dNE%dNC%d/Crosslink_distribution_initial.txt",
            slope, INITIAL_FILE, SAMPLE_NUMBER, Num_crosslink, Thresh_D_S, F_0_CRMT, Num_motor, Num_Ext, Num_Con);
    std::ofstream outfile(filename);
    //outfile << "Distance_from_center\tNumber_of_crosslinks\tNormalized_density\n";
    for (const auto& entry : crosslink_distribution) {
        double distance = entry.first;
        int count = entry.second;
        int shell_index = std::min(static_cast<int>(distance / shell_width), NUM_SHELLS - 1);
        double normalized_density = shell_densities[shell_index] / max_density;
        outfile << distance << "\t" << count << "\t" << normalized_density << "\n";
        }
    outfile.close();
}

void createShellConnections(int N_Cell_Lamin, int Beads_per_Chain) {
    std::default_random_engine generator(time(NULL) + distribution_uni_seed(generator_seed));
    std::uniform_real_distribution<double> distribution(0.0, 1.0);
    
    // First, identify potential connection points on both shells
    std::vector<int> inner_shell_candidates;
    std::vector<int> outer_shell_candidates;
    
    // Find inner shell monomers that aren't already connected to the outer shell
    for (int i = SYSTEMSIZE; i < SYSTEMSIZE + SYSTEMSIZE2; i++) {
        bool already_connected = false;
        for (size_t j = 0; j < Monomer[i].springs.size(); j++) {
            if (Monomer[i].springs[j] >= SYSTEMSIZE + SYSTEMSIZE2) {
                already_connected = true;
                break;
            }
        }
        if (!already_connected) {
            inner_shell_candidates.push_back(i);
        }
    }
    
    // Find outer shell monomers that aren't already connected to the inner shell
    for (int i = SYSTEMSIZE + SYSTEMSIZE2; i < SYSTEMSIZE + SYSTEMSIZE2 + SYSTEMSIZE3; i++) {
        bool already_connected = false;
        for (size_t j = 0; j < Monomer[i].springs.size(); j++) {
            if (Monomer[i].springs[j] >= SYSTEMSIZE && Monomer[i].springs[j] < SYSTEMSIZE + SYSTEMSIZE2) {
                already_connected = true;
                break;
            }
        }
        if (!already_connected) {
            outer_shell_candidates.push_back(i);
        }
    }
    
    // Shuffle candidates for randomness
    std::shuffle(inner_shell_candidates.begin(), inner_shell_candidates.end(), generator);
    std::shuffle(outer_shell_candidates.begin(), outer_shell_candidates.end(), generator);
    
    // Determine how many connections we can actually make
    int possible_connections = std::min(
        std::min(static_cast<int>(inner_shell_candidates.size()), 
                static_cast<int>(outer_shell_candidates.size())),
        N_Cell_Lamin
    );
    
    if (possible_connections == 0) {
        std::cout << "Warning: Cannot create any shell-to-shell connections due to lack of available monomers." << std::endl;
        return;
    }
    
    // Create a vector to store the indices of the new intermediate beads
    std::vector<int> new_beads;
    int current_size = SYSTEMSIZE + SYSTEMSIZE2 + SYSTEMSIZE3;
    
    // For each connection
    for (int conn = 0; conn < possible_connections; conn++) {
        int inner_monomer = inner_shell_candidates[conn];
        int outer_monomer = outer_shell_candidates[conn];
        
        // Calculate the straight line between the two monomers
        double dx = Monomer[outer_monomer].x - Monomer[inner_monomer].x;
        double dy = Monomer[outer_monomer].y - Monomer[inner_monomer].y;
        double dz = Monomer[outer_monomer].z - Monomer[inner_monomer].z;
        double total_dist = std::sqrt(dx*dx + dy*dy + dz*dz);
        
        // Create intermediate beads
        std::vector<int> chain_beads;
        chain_beads.push_back(inner_monomer);
        
        for (int b = 0; b < Beads_per_Chain; b++) {
            // Create a new monomer for the intermediate bead
            Monomer[current_size].node = current_size;
            Monomer[current_size].r = r_CRMT;  // Use same radius as CRMT monomers
            Monomer[current_size].k = k_CRMT;  // Use same stiffness as CRMT
            Monomer[current_size].status = 2;  // Special status for linker beads
            
            // Position the bead along the straight line
            double fraction = (b + 1.0) / (Beads_per_Chain + 1.0);
            Monomer[current_size].x = Monomer[inner_monomer].x + dx * fraction;
            Monomer[current_size].y = Monomer[inner_monomer].y + dy * fraction;
            Monomer[current_size].z = Monomer[inner_monomer].z + dz * fraction;
            
            // Initialize hash value
            Monomer[current_size].hashvalue = int((Monomer[current_size].x + 0.5*MaxMesh)/Mesh_size) + 
                                             Mesh_num*int((Monomer[current_size].y + 0.5*MaxMesh)/Mesh_size) + 
                                             Mesh_num*Mesh_num*int((Monomer[current_size].z + 0.5*MaxMesh)/Mesh_size);
            Monomer[current_size].next = HashchartHead[Monomer[current_size].hashvalue];
            HashchartHead[Monomer[current_size].hashvalue] = &(Monomer[current_size]);
            
            chain_beads.push_back(current_size);
            new_beads.push_back(current_size);
            current_size++;
        }
        
        chain_beads.push_back(outer_monomer);
        
        // Create springs between consecutive beads in the chain
        for (size_t i = 0; i < chain_beads.size() - 1; i++) {
            int bead1 = chain_beads[i];
            int bead2 = chain_beads[i+1];
            
            // Calculate rest length
            double rest_length = std::sqrt(
                std::pow(Monomer[bead1].x - Monomer[bead2].x, 2) +
                std::pow(Monomer[bead1].y - Monomer[bead2].y, 2) +
                std::pow(Monomer[bead1].z - Monomer[bead2].z, 2)
            );
            
            // Create bidirectional springs
            Monomer[bead1].springs.push_back(bead2);
            Monomer[bead1].a_s.push_back(rest_length);
            Monomer[bead1].k_s.push_back(k_CRMT);
            
            Monomer[bead2].springs.push_back(bead1);
            Monomer[bead2].a_s.push_back(rest_length);
            Monomer[bead2].k_s.push_back(k_CRMT);
        }
    }
    
    // Log the connections
    char filename[400];
    sprintf(filename, "Soft_Cell_Nuclei_M%.2lf/INITIAL%dSAMPLE%dXLINK%dLAD%.2lfF%.1lfNM%dNE%dNC%d/Shells_connections.txt",
            slope, INITIAL_FILE, SAMPLE_NUMBER, Num_crosslink, Thresh_D_S, F_0_CRMT, Num_motor, Num_Ext, Num_Con);
    
    std::ofstream outfile(filename);
    outfile << "Created " << possible_connections << " shell-to-shell connections with " 
            << Beads_per_Chain << " intermediate beads each." << std::endl;
    outfile << "Total new beads added: " << new_beads.size() << std::endl;
    outfile << "New system size: " << current_size << std::endl;
    outfile.close();
}

void calculateRadialCrosslinkDistribution() {
    std::vector<double> center = calculateShellCenter();
    const int NUM_BINS = 80;  // Number of bins for radial distribution
    std::vector<int> crosslink_counts(NUM_BINS, 0);
    std::vector<int> particle_counts(NUM_BINS, 0);
    double max_distance = 0.0;

    // First pass: calculate max distance
    for (int i = 0; i < SYSTEMSIZE; i++) {
        double dx = Monomer[i].x - center[0];
        double dy = Monomer[i].y - center[1];
        double dz = Monomer[i].z - center[2];
        double distance = sqrt(dx*dx + dy*dy + dz*dz);
        max_distance = std::max(max_distance, distance);
    }

    // Second pass: count crosslinks and particles in bins
    for (int i = 0; i < SYSTEMSIZE; i++) {
        double dx = Monomer[i].x - center[0];
        double dy = Monomer[i].y - center[1];
        double dz = Monomer[i].z - center[2];
        double distance = sqrt(dx*dx + dy*dy + dz*dz);
        int bin = std::min(static_cast<int>(NUM_BINS * distance / max_distance), NUM_BINS - 1);
        particle_counts[bin]++;
        for (size_t j = 0; j < Monomer[i].springs.size(); j++) {
            if (Monomer[i].springs[j] > i && Monomer[i].springs[j] < SYSTEMSIZE) {
                crosslink_counts[bin]++;
            }
        }
    }

    // Save the binned crosslink distribution to a file
    char filename[400];
    sprintf(filename, "Soft_Cell_Nuclei_M%.2lf/INITIAL%dSAMPLE%dXLINK%dLAD%.2lfF%.1lfNM%dNE%dNC%d/Crosslink_distribution_t%.0lf.txt",
            slope, INITIAL_FILE, SAMPLE_NUMBER, Num_crosslink, Thresh_D_S, F_0_CRMT, Num_motor, Num_Ext, Num_Con, running_time);
    std::ofstream outfile(filename);
    outfile << "Distance BinCenter ParticleCount CrosslinkCount Density CrosslinksPerParticle TotalCrosslinks NumberDensity\n";
    double bin_width = max_distance / NUM_BINS;
    int total_crosslinks = std::accumulate(crosslink_counts.begin(), crosslink_counts.end(), 0);

    for (int i = 0; i < NUM_BINS; i++) {
        double bin_center = (i + 0.5) * bin_width;
        double distance = i * bin_width;
        int particle_count = particle_counts[i];
        int crosslink_count = crosslink_counts[i];
        double density = static_cast<double>(crosslink_count) / total_crosslinks;
        double crosslinks_per_particle = particle_count > 0 ? static_cast<double>(crosslink_count) / particle_count : 0;
        
        // Calculate the volume of the spherical shell for this bin
        double outer_radius = (i + 1) * bin_width;
        double inner_radius = i * bin_width;
        double shell_volume = (4.0/3.0) * M_PI * (std::pow(outer_radius, 3) - std::pow(inner_radius, 3));

        // Calculate number density (crosslinks per unit volume)
        double number_density_CL = shell_volume > 0 ? static_cast<double>(crosslink_count) / shell_volume : 0;

        outfile << distance << " " << bin_center << " " << particle_count << " " 
                << crosslink_count << " " << density << " " << crosslinks_per_particle << " " 
                << total_crosslinks << " " << number_density_CL <<"\n";
    }

    outfile.close();
}

void calculateRadialMotorDistribution() {
	if (!MOTOR_IS_ON) return;  // Exit the function if motors are not active
	
    std::vector<double> center = calculateShellCenter();
    const int NUM_BINS = 80;  // Number of bins for radial distribution
    std::vector<int> motor_counts(NUM_BINS, 0);
    std::vector<int> particle_counts(NUM_BINS, 0);
    double max_distance = 0.0;

    // First pass: calculate max distance
    for (int i = 0; i < SYSTEMSIZE; i++) {
        double dx = Monomer[i].x - center[0];
        double dy = Monomer[i].y - center[1];
        double dz = Monomer[i].z - center[2];
        double distance = sqrt(dx*dx + dy*dy + dz*dz);
        max_distance = std::max(max_distance, distance);
    }

    // Second pass: count motors and particles in bins
    for (int i = 0; i < SYSTEMSIZE; i++) {
        double dx = Monomer[i].x - center[0];
        double dy = Monomer[i].y - center[1];
        double dz = Monomer[i].z - center[2];
        double distance = sqrt(dx*dx + dy*dy + dz*dz);
        int bin = std::min(static_cast<int>(NUM_BINS * distance / max_distance), NUM_BINS - 1);
        particle_counts[bin]++;
        if (Monomer[i].status != 0) {  // If it's a motor
            motor_counts[bin]++;
        }
    }

    // Save the binned motor distribution to a file
    char filename[400];
    sprintf(filename, "Soft_Cell_Nuclei_M%.2lf/INITIAL%dSAMPLE%dXLINK%dLAD%.2lfF%.1lfNM%dNE%dNC%d/Motor_distribution_t%.0lf.txt",
            slope, INITIAL_FILE, SAMPLE_NUMBER, Num_crosslink, Thresh_D_S, F_0_CRMT, Num_motor, Num_Ext, Num_Con, running_time);
    std::ofstream outfile(filename);

    double bin_width = max_distance / NUM_BINS;
    int total_motors = std::accumulate(motor_counts.begin(), motor_counts.end(), 0);

    for (int i = 0; i < NUM_BINS; i++) {
        double bin_center = (i + 0.5) * bin_width;
        double distance = i * bin_width;
        int particle_count = particle_counts[i];
        int motor_count = motor_counts[i];
        double density = static_cast<double>(motor_count) / total_motors;
        double motors_per_particle = particle_count > 0 ? static_cast<double>(motor_count) / particle_count : 0;
        
        // Calculate the volume of the spherical shell for this bin
        double outer_radius = (i + 1) * bin_width;
        double inner_radius = i * bin_width;
        double shell_volume = (4.0/3.0) * M_PI * (std::pow(outer_radius, 3) - std::pow(inner_radius, 3));

        // Calculate number density (motors per unit volume)
        double number_density_motors = shell_volume > 0 ? static_cast<double>(motor_count) / shell_volume : 0;
        
        outfile << distance << " " << bin_center << " " << particle_count << " " 
                << motor_count << " " << density << " " << motors_per_particle << " " 
                << total_motors << " " << number_density_motors <<"\n";
    }
    outfile.close();
}

void Initialize(int Add_on_random_time){
    int i,j,n;
    std::default_random_engine generator(time(NULL)+Add_on_random_time);
    std::uniform_int_distribution<long long> distribution_uni(0,50000000);

    char filename_initial[100];
    sprintf (filename_initial, "Configurations/Final_Config%d.txt",INITIAL_FILE);
    std::ifstream file_Monomer;
    file_Monomer.open(filename_initial);
    for(n=0;n<SYSTEMSIZE;n++)
    {
        file_Monomer >> Monomer[n].x >> Monomer[n].y >> Monomer[n].z >> Monomer[n].status;
    }
    file_Monomer.close();
    
    // read in Nuclear shell monomer's positions
    char filename_shell[100];
    sprintf(filename_shell, "Nuclei_Shell/%d_final_shell_position.txt",INITIAL_FILE);
    std::ifstream file_shell;
    file_shell.open(filename_shell);
    for(n=SYSTEMSIZE;n<SYSTEMSIZE+SYSTEMSIZE2;n++)
    {
        file_shell >> Monomer[n].x >> Monomer[n].y >> Monomer[n].z >> Monomer[n].status;
//        cout << Monomer[n].z << endl;
    }
    file_shell.close();
    
    // read in shell monomer network from randomization
    char filename_link[50];
    sprintf(filename_link, "Nuclei_Shell/%d_final_shell_links.txt", INITIAL_FILE);
    std::ifstream file_shell2;
    file_shell2.open(filename_link);
    int I, J;
    double rest_length;
    
    int total_links, temp2;
    double temp3;
    
    file_shell2>> total_links >> temp2 >> temp3;
    for (n=0;n<total_links;n++)
    {
   //   cout << I << " " << J << " " << rest_length << endl;
        file_shell2 >> I >> J >> rest_length;
        Monomer[SYSTEMSIZE+I].springs.push_back(SYSTEMSIZE+J);
        Monomer[SYSTEMSIZE+I].a_s.push_back(rest_length);
        Monomer[SYSTEMSIZE+I].k_s.push_back(k_shell);
        Monomer[SYSTEMSIZE+J].springs.push_back(SYSTEMSIZE+I);
        Monomer[SYSTEMSIZE+J].a_s.push_back(rest_length);
        Monomer[SYSTEMSIZE+J].k_s.push_back(k_shell);
    }
    file_shell2.close();

    // Read outer shell monomer positions
    char filename_outershell[100];
    sprintf(filename_outershell, "Cell_Shell/%d_final_outershell_position.txt", INITIAL_FILE);
    std::ifstream file_outershell;
    file_outershell.open(filename_outershell);
    for(n=SYSTEMSIZE+SYSTEMSIZE2; n<SYSTEMSIZE+SYSTEMSIZE2+SYSTEMSIZE3; n++) {
        file_outershell >> Monomer[n].x >> Monomer[n].y >> Monomer[n].z >> Monomer[n].status;
    }
    file_outershell.close();
    
    // Read outer shell network links
    char filename_outershell_link[50];
    sprintf(filename_outershell_link, "Cell_Shell/%d_final_outershell_links.txt", INITIAL_FILE);
    std::ifstream file_outershell2;
    file_outershell2.open(filename_outershell_link);
    
    int total_outer_links, temp2_outer;
    double temp3_outer;
    file_outershell2 >> total_outer_links >> temp2_outer >> temp3_outer;
    
    for(n=0; n<total_outer_links; n++) {
        int I_outer, J_outer;
        double rest_length_outer;
        file_outershell2 >> I_outer >> J_outer >> rest_length_outer;
        
        Monomer[SYSTEMSIZE+SYSTEMSIZE2+I_outer].springs.push_back(SYSTEMSIZE+SYSTEMSIZE2+J_outer);
        Monomer[SYSTEMSIZE+SYSTEMSIZE2+I_outer].a_s.push_back(rest_length_outer);
        Monomer[SYSTEMSIZE+SYSTEMSIZE2+I_outer].k_s.push_back(k_outershell);
        
        Monomer[SYSTEMSIZE+SYSTEMSIZE2+J_outer].springs.push_back(SYSTEMSIZE+SYSTEMSIZE2+I_outer);
        Monomer[SYSTEMSIZE+SYSTEMSIZE2+J_outer].a_s.push_back(rest_length_outer);
        Monomer[SYSTEMSIZE+SYSTEMSIZE2+J_outer].k_s.push_back(k_outershell);
    }
    file_outershell2.close();    
    //  cout << "done" << endl;
    
    // Initialize the monomer properties
    for(n=0;n<SYSTEMSIZE;n++){
        Monomer[n].node=n;
        Monomer[n].r=r_CRMT;
        Monomer[n].k=k_CRMT;
        Monomer[n].status=-1;
        Monomer[n].next=nullptr;
    }
    
    for(n=SYSTEMSIZE;n<SYSTEMSIZE+SYSTEMSIZE2;n++){
        Monomer[n].node=n;
        Monomer[n].r=r_shell;
        Monomer[n].k=k_shell;
        Monomer[n].status=-1;
        Monomer[n].next=nullptr;
    }
    
    for(n=SYSTEMSIZE+SYSTEMSIZE2; n<SYSTEMSIZE+SYSTEMSIZE2+SYSTEMSIZE3; n++) {
        Monomer[n].node = n;
        Monomer[n].r = r_outershell;
        Monomer[n].k = k_outershell;
        Monomer[n].status = 3;
        Monomer[n].next = nullptr;
    }
       
    // initialize hash value for each shell monomer
    for(n=0;n<SYSTEMSIZE+SYSTEMSIZE2+SYSTEMSIZE3;n++){
        Monomer[n].hashvalue=int((Monomer[n].x+0.5*MaxMesh)/Mesh_size)+Mesh_num*int((Monomer[n].y+0.5*MaxMesh)/Mesh_size)+Mesh_num*Mesh_num*int((Monomer[n].z+0.5*MaxMesh)/Mesh_size);
        Monomer[n].next=HashchartHead[Monomer[n].hashvalue];
        HashchartHead[Monomer[n].hashvalue]=&(Monomer[n]);
    }
   
    // Initialize crosslinks and cell-nuceli linkages
    createCrosslinks(Num_crosslink, Thresh);
    createShellConnections(N_Cell_Lamin, Beads_per_Chain);
       
    // initialize LAD links (here status of polymer is changed, need to set to zero after this)
    double distance;
    for(j=SYSTEMSIZE;j<SYSTEMSIZE+SYSTEMSIZE2;j++){
        double temp=Thresh_D_S;
        int temp_ind=-1;
        for(i=0;i<SYSTEMSIZE;i++){
            if( Monomer[i].status==-1 ){
                distance=sqrt(pow(Monomer[i].x-Monomer[j].x,2)+pow(Monomer[i].y-Monomer[j].y,2)+pow(Monomer[i].z-Monomer[j].z,2));
                if(distance<temp){
                    temp=distance;
                    temp_ind=i;
                }
            }
        }
        if( temp_ind>-1 ){
            Monomer[temp_ind].status=j;
            Monomer[temp_ind].springs.push_back(j);
            Monomer[temp_ind].a_s.push_back(temp);
            Monomer[temp_ind].k_s.push_back(k_D_S);    //Monomer[temp_ind].k_s.push_back(k_crosslink);
            Monomer[j].status=temp_ind;
            Monomer[j].springs.push_back(temp_ind);
            Monomer[j].a_s.push_back(temp);
            Monomer[j].k_s.push_back(k_D_S);    //Monomer[j].k_s.push_back(k_crosslink);
        }
    }
    for(n=0;n<SYSTEMSIZE;n++){ // restore CRMT monomer status
        Monomer[n].status=0;
    }
//    cout << "LAD links completed!" << endl;
    
    // initialize motors
    if(MOTOR_IS_ON){
        int randlist[SYSTEMSIZE];
        for(i=0;i<SYSTEMSIZE;i++){
            randlist[i]=i;
        }
        for(i=0;i<Num_motor;i++){
            int bbb;
            int aaa=int(distribution_uni(generator)%(SYSTEMSIZE-i));
            bbb=randlist[i+aaa];
            randlist[i+aaa]=randlist[i];
            randlist[i]=bbb;
            if (i<Num_Con){Monomer[bbb].status=-1;}
            else {Monomer[bbb].status=1;}
        }
    }
}

void writeErrorToFile(const std::string& message) {
    char filename[400];
    sprintf(filename, "Soft_Cell_Nuclei_M%.2lf/INITIAL%dSAMPLE%dXLINK%dLAD%.2lfF%.1lfNM%dNE%dNC%d/error_log.txt",
            slope, INITIAL_FILE, SAMPLE_NUMBER, Num_crosslink, Thresh_D_S, F_0_CRMT, Num_motor, Num_Ext, Num_Con);
    
    std::ofstream errorFile(filename, std::ios_base::app);  // Open in append mode
    if (errorFile.is_open()) {
        errorFile << "Error at time " << running_time << ": " << message << std::endl;
        errorFile.close();
    } else {
        std::cerr << "Failed to open error log file." << std::endl;
    }
}

void writeLAMMPSData(const char* filename) {
    std::ofstream outfile(filename);
    outfile << std::setprecision(8);
    int total_particles = SYSTEMSIZE + SYSTEMSIZE2 + SYSTEMSIZE3;

    // Debug counters
    int polymer_bonds = SYSTEMSIZE - 1;
    int crosslinks = 0;
    int lad_links = 0;
    int shell_bonds = 0;

    // Count crosslinks, LAD links, and shell bonds
    for (int i = 0; i < SYSTEMSIZE + SYSTEMSIZE2; i++) {
        for (size_t j = 0; j < Monomer[i].springs.size(); j++) {
            if (Monomer[i].springs[j] > i) {
                if (i < SYSTEMSIZE && Monomer[i].springs[j] < SYSTEMSIZE) {
                    crosslinks++;
                } else if (i >= SYSTEMSIZE && Monomer[i].springs[j] >= SYSTEMSIZE) {
                    shell_bonds++;
                }
            }
        }
        if (i >= SYSTEMSIZE && Monomer[i].status != -1) {
            lad_links++;
        }
    }

    int total_bonds = polymer_bonds + crosslinks + lad_links + shell_bonds;

    // Header
    outfile << "LAMMPS data file for CRMT, Shell, and Motor system\n\n";
    outfile << total_particles << " atoms\n";
    outfile << total_bonds << " bonds\n";
    outfile << "#" << lad_links << " LADs\n\n";

    // Determine the number of atom types
    bool has_extensile = false;
    bool has_contractile = false;
    for (int i = 0; i < SYSTEMSIZE; i++) {
        if (Monomer[i].status == 1) has_extensile = true;
        if (Monomer[i].status == -1) has_contractile = true;
        if (has_extensile && has_contractile) break;
    }

    int atom_types = 2;  // CRMT and Shell are always present
    if (has_extensile) atom_types++;
    if (has_contractile) atom_types++;
        
    outfile << atom_types << " atom types\n";  // CRMT, Shell, Motors (E and C respectively)    
    outfile << "4 bond types\n\n";  // Polymer, Crosslink, LAD, Shell

    // Box dimensions
    outfile << -MaxMesh/2 << " " << MaxMesh/2 << " xlo xhi\n";
    outfile << -MaxMesh/2 << " " << MaxMesh/2 << " ylo yhi\n";
    outfile << -MaxMesh/2 << " " << MaxMesh/2 << " zlo zhi\n\n";

    // Atoms section
    outfile << "Atoms\n\n";
    for (int i = 0; i < total_particles; i++) {
        int type;
        if (i < SYSTEMSIZE) {
            if (Monomer[i].status == 0) {
                type = 1;  // Regular CRMT monomer
            } else if (Monomer[i].status == 1) {
                type = 3;  // Extensile motor
            } else if (Monomer[i].status == -1) {
                type = has_extensile ? 4 : 3;  // Contractile motor (type 4 if Extensile is present else type 3)
            } else {
                type = 1;  // Default to regular CRMT if status is unexpected
            }
        } else {
            type = 2;  // Shell monomer
        }
        
        outfile << i+1 << " " << type << " " << type << " "  //atom_id mol_id atom_type (mol_id ==atom_type)
                << Monomer[i].x << " " << Monomer[i].y << " " << Monomer[i].z << "\n";
    }

    outfile << "\n";

    // Bonds section
    outfile << "Bonds\n\n";
    int bond_id = 1;
    int written_polymer_bonds = 0;
    int written_crosslinks = 0;
    int written_lad_links = 0;
    int written_shell_bonds = 0;

    // 1. Polymer bonds
    for (int i = 0; i < SYSTEMSIZE - 1; i++) {
        outfile << bond_id++ << " 1 " << i+1 << " " << i+2 << "\n";
        written_polymer_bonds++;
    }

    // 2. Crosslinks
    for (int i = 0; i < SYSTEMSIZE; i++) {
        for (size_t j = 0; j < Monomer[i].springs.size(); j++) {
            if (Monomer[i].springs[j] > i && Monomer[i].springs[j] < SYSTEMSIZE) {
                outfile << bond_id++ << " 2 " << i+1 << " " << Monomer[i].springs[j]+1 << "\n";
                written_crosslinks++;
            }
        }
    }

    // 3. LAD links
    for (int i = SYSTEMSIZE; i < SYSTEMSIZE + SYSTEMSIZE2; i++) {
        if (Monomer[i].status != -1) {
            outfile << bond_id++ << " 3 " << Monomer[i].status+1 << " " << i+1 << "\n";
            written_lad_links++;
        }
    }

    // 4. Shell bonds
    for (int i = SYSTEMSIZE; i < SYSTEMSIZE + SYSTEMSIZE2; i++) {
        for (size_t j = 0; j < Monomer[i].springs.size(); j++) {
            if (Monomer[i].springs[j] > i) {
                outfile << bond_id++ << " 4 " << i+1 << " " << Monomer[i].springs[j]+1 << "\n";
                written_shell_bonds++;
            }
        }
    }

    outfile.close();

    // Verification
    if (bond_id - 1 != total_bonds) {
		std::stringstream ss;
		ss << "Mismatch in bond count. Header: " << total_bonds 
		   << ", Written: " << (bond_id - 1) << std::endl
		   << "Polymer bonds: expected " << polymer_bonds << ", written " << written_polymer_bonds << std::endl
		   << "Crosslinks: expected " << crosslinks << ", written " << written_crosslinks << std::endl
		   << "LAD links: expected " << lad_links << ", written " << written_lad_links << std::endl
		   << "Shell bonds: expected " << shell_bonds << ", written " << written_shell_bonds;
		writeErrorToFile(ss.str());
    }

}

void writeLAMMPSTRJ(const char* filename, bool append, double sim_time) {
    std::ofstream outfile;
    if (append) {
        outfile.open(filename, std::ios_base::app);
    } else {
        outfile.open(filename);
    }

    outfile << std::scientific << std::setprecision(8);

    int total_particles = SYSTEMSIZE + SYSTEMSIZE2;

    // Determine the number of atom types
    bool has_extensile = false;
    bool has_contractile = false;
    for (int i = 0; i < SYSTEMSIZE; i++) {
        if (Monomer[i].status == 1) has_extensile = true;
        if (Monomer[i].status == -1) has_contractile = true;
        if (has_extensile && has_contractile) break;
    }
    
    outfile << "ITEM: TIMESTEP\n";
    outfile << static_cast<long long>(sim_time / timestep) << "\n";  // Convert to timestep number
    outfile << "ITEM: NUMBER OF ATOMS\n";
    outfile << total_particles << "\n";
    outfile << "ITEM: BOX BOUNDS pp pp pp\n";
    outfile << -MaxMesh/2 << " " << MaxMesh/2 << "\n";
    outfile << -MaxMesh/2 << " " << MaxMesh/2 << "\n";
    outfile << -MaxMesh/2 << " " << MaxMesh/2 << "\n";
    outfile << "ITEM: ATOMS id type x y z ix iy iz\n";

    for (int i = 0; i < total_particles; i++) {
        int type;
        if (i < SYSTEMSIZE) {
            if (Monomer[i].status == 0) {
                type = 1;  // Regular CRMT monomer
            } else if (Monomer[i].status == 1) {
                type = 3;  // Extensile motor
            } else if (Monomer[i].status == -1) {
                type = has_extensile ? 4 : 3;  // Contractile motor (type 4 if Extensile is present else type 3)
            } else {
                type = 1;  // Default to regular CRMT if status is unexpected
            }
        } else {
            type = 2;  // Shell monomer
        }

        outfile << i+1 << " " << type << " "
                << Monomer[i].x << " " << Monomer[i].y << " " << Monomer[i].z 
                << " 0 0 0\n";  // Adding image flags (ix iy iz)
    }

    outfile.close();
}

void writeSummaryFile() {
    char filename[400];
    sprintf(filename, "Soft_Cell_Nuclei_M%.2lf/INITIAL%dSAMPLE%dXLINK%dLAD%.2lfF%.1lfNM%dNE%dNC%d/Summary_t%.0f.txt",
            slope, INITIAL_FILE, SAMPLE_NUMBER, Num_crosslink, Thresh_D_S, F_0_CRMT, Num_motor, Num_Ext, Num_Con, running_time);
    
    std::ofstream outfile(filename);
    
    // Count total crosslinks
    int total_crosslinks = 0;
    for (int i = 0; i < SYSTEMSIZE; i++) {
        for (size_t j = 0; j < Monomer[i].springs.size(); j++) {
            if (Monomer[i].springs[j] > i && Monomer[i].springs[j] < SYSTEMSIZE) {
                total_crosslinks++;
            }
        }
    }
    
    // Count number of motors
    int num_motors = 0;
    for (int i = 0; i < SYSTEMSIZE; i++) {
        if (Monomer[i].status != 0) {
            num_motors++;
        }
    }
    
    // Count LADs (polymer-shell linkages)
    int num_lads = 0;
    for (int i = SYSTEMSIZE; i < SYSTEMSIZE + SYSTEMSIZE2; i++) {
        if (Monomer[i].status != -1) {
            num_lads++;
        }
    }

    // Count motors in different regions
    int inner_motors = 0, middle_motors = 0, outer_motors = 0;
    std::vector<double> center = calculateShellCenter();
    double max_distance = 0;
    
    // First pass to find max distance
    for (int i = 0; i < SYSTEMSIZE; i++) {
        double dx = Monomer[i].x - center[0];
        double dy = Monomer[i].y - center[1];
        double dz = Monomer[i].z - center[2];
        double distance = sqrt(dx*dx + dy*dy + dz*dz);
        max_distance = std::max(max_distance, distance);
    }

    // Second pass to count motors in regions
    for (int i = 0; i < SYSTEMSIZE; i++) {
        if (Monomer[i].status != 0) {  // If it's a motor
            double dx = Monomer[i].x - center[0];
            double dy = Monomer[i].y - center[1];
            double dz = Monomer[i].z - center[2];
            double distance = sqrt(dx*dx + dy*dy + dz*dz);
            
            if (distance < max_distance / 3) inner_motors++;
            else if (distance < 2 * max_distance / 3) middle_motors++;
            else outer_motors++;
        }
    }
        
    outfile << "Simulation Summary at t = " << running_time << std::endl;
    outfile << "Total Crosslinks: " << total_crosslinks << std::endl;
    outfile << "Total Number of Motors: " << num_motors << std::endl;
    outfile << "Number of E_Motors: " << num_motors-Num_Con << std::endl;
    outfile << "Number of C_Motors: " << Num_Con << std::endl;
    outfile << "Thresh_D LADs: " << Thresh_D_S << std::endl;
    outfile << "Number of LADs: " << num_lads << std::endl;
    outfile << "Motors in inner third: " << inner_motors << std::endl;
    outfile << "Motors in middle third: " << middle_motors << std::endl;
    outfile << "Motors in outer third: " << outer_motors << std::endl;    
    outfile.close();
}

        
int main(int argc, char **argv){
    int i,j,k;
    // set up parameters
    INITIAL_FILE=atoi(argv[1]); // 0~101
    int param_xlink=atoi(argv[2]); // 0,1,2,3,4,5
    int param_motor=atoi(argv[3]); // 0,1,2,3,4
    int param_binding=atoi(argv[4]); // 0,1,2,3,4
    int param_run=atoi(argv[5]); // 0
    int param_slink=atoi(argv[6]); //0~7
    int param_bpc=atoi(argv[7]); //0~4
    
    Num_crosslink=List_Num_xlink[param_xlink];
    F_0_CRMT=List_F0[param_motor];
    N_Cell_Lamin=List_Num_Shelllinks[param_slink];
    Beads_per_Chain=List_BeadsperChain[param_bpc];
    
    if(INITIAL_FILE<100){
        Thresh_D_S=List_Thresh_D_S[param_binding];
    }
    else{
        Thresh_D_S=List_specialtwo[INITIAL_FILE-100][param_binding];
    }
    
    SAMPLE_NUMBER+=(6*param_binding+param_xlink);

    if(param_motor==0){MOTOR_IS_ON=false;}
    else{MOTOR_IS_ON=true;}
    
    Num_crosslink=List_Num_xlink[param_xlink];

    bool first_step_completed = false;
    
    char mainfolder[100];
    sprintf (mainfolder, "Soft_Cell_Nuclei_M%.2lf",slope);
    mkdir(mainfolder, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);    
    
    char subfolder[400];
    sprintf (subfolder, "Soft_Cell_Nuclei_M%.2lf/INITIAL%dSAMPLE%dXLINK%dLAD%.2lfF%.1lfNM%dNE%dNC%d",slope,INITIAL_FILE,SAMPLE_NUMBER,Num_crosslink,Thresh_D_S,F_0_CRMT,Num_motor,Num_Ext,Num_Con);
    mkdir(subfolder, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    
    // Initialization
    Initialize(SAMPLE_NUMBER*1000+INITIAL_FILE*100+param_run*10+distribution_uni_seed(generator_seed)); 
    
    calculateRadialCrosslinkDistribution();  // Record cross link at t=0
    
    if (MOTOR_IS_ON) {
    calculateRadialMotorDistribution(); // Record motor distribution at t=0
    }
    //    cout << "initialization completed!" << endl;   
       
    char filename[400];
    sprintf(filename, "Soft_Cell_Nuclei_M%.2lf/INITIAL%dSAMPLE%dXLINK%dLAD%.2lfF%.1lfNM%dNE%dNC%d/Simulation_Cell_Nuclei_Traj.lammpstrj",slope,INITIAL_FILE, SAMPLE_NUMBER, Num_crosslink, Thresh_D_S, F_0_CRMT, Num_motor, Num_Ext, Num_Con);
    int step = 0;  
    
    const double TIME_TOLERANCE = 0.5 * timestep; 

    // run the simulation and output every 1sec
    while(running_time<total_time){
        if((int(running_time/timestep)%Plot_cycle)==0){
            char filename_Monomer[400];
            sprintf (filename_Monomer, "Soft_Cell_Nuclei_M%.2lf/INITIAL%dSAMPLE%dXLINK%dLAD%.2lfF%.1lfNM%dNE%dNC%d/t=%lf_CRMT_xyz.txt",slope,INITIAL_FILE,SAMPLE_NUMBER,Num_crosslink,Thresh_D_S,F_0_CRMT,Num_motor,Num_Ext,Num_Con,Plot_label);
            std::ofstream outfile_Monomer(filename_Monomer);
            for(k=0;k<SYSTEMSIZE;k++){
                outfile_Monomer << Monomer[k].x << " " << Monomer[k].y << " " << Monomer[k].z << " " << Monomer[k].status << std::endl;
            }
            if(SHELL_IS_SOFT){
                for(k=SYSTEMSIZE;k<SYSTEMSIZE+SYSTEMSIZE2;k++){
                    outfile_Monomer << Monomer[k].x << " " << Monomer[k].y << " " << Monomer[k].z << " " << Monomer[k].status << std::endl;
                }
            }
            outfile_Monomer.close();

            Plot_label+=(Plot_cycle*timestep);
        }
        clearF(0,SYSTEMSIZE+SYSTEMSIZE2+SYSTEMSIZE3);
        Polymer_spring();
        Other_spring(0,SYSTEMSIZE+SYSTEMSIZE2+SYSTEMSIZE3);
//        Boundary1_Constraints(0,SYSTEMSIZE);
//        Spherical_Constraint(SYSTEMSIZE,SYSTEMSIZE+SYSTEMSIZE2);
        ThermalFluctuation(running_time*10000+SAMPLE_NUMBER*1000+INITIAL_FILE*100+param_run*10+distribution_uni_seed(generator_seed)+int(running_time*100000),0,SYSTEMSIZE+SYSTEMSIZE2+SYSTEMSIZE3);
        UpdateSoftRepulsion(0,SYSTEMSIZE+SYSTEMSIZE2+SYSTEMSIZE3);
        if(MOTOR_IS_ON){UpdateMotorForce(0,SYSTEMSIZE);}
        if(MOTOR_IS_ON && (int(running_time/timestep)%(UpdateMotorCycle))==0){
            Updatemotor(running_time*10000+SAMPLE_NUMBER*1000+INITIAL_FILE*100+param_run*10+distribution_uni_seed(generator_seed)+int(running_time*100000));
        }
        Updatexyz(0,SYSTEMSIZE+SYSTEMSIZE2+SYSTEMSIZE3);
        UpdateHash(0,SYSTEMSIZE+SYSTEMSIZE2+SYSTEMSIZE3);
        //running_time+=timestep;        
        
        if (!first_step_completed) {
            char filename[400];
            sprintf(filename, "Soft_Cell_Nuclei_M%.2lf/INITIAL%dSAMPLE%dXLINK%dLAD%.2lfF%.1lfNM%dNE%dNC%d/Lammps_Data_Cell_Nuclei_Initial.dat",slope,INITIAL_FILE, SAMPLE_NUMBER, Num_crosslink, Thresh_D_S, F_0_CRMT, Num_motor, Num_Ext, Num_Con);
            writeLAMMPSData(filename);
            first_step_completed = true;
        }
        
        // Check for specific time to dump the crosslink data
        if (abs(running_time - 1.0) < TIME_TOLERANCE ||
            std::abs(running_time - 250.0) < TIME_TOLERANCE ||
            std::abs(running_time - 500.0) < TIME_TOLERANCE ||
            std::abs(running_time - 750.0) < TIME_TOLERANCE ||
            std::abs(running_time - 999.0) < TIME_TOLERANCE) { 
            
            calculateRadialCrosslinkDistribution();
            if (MOTOR_IS_ON){
				calculateRadialMotorDistribution();
			} 
            writeSummaryFile();
		}        
        
        if ((int(running_time/timestep) % Plot_cycle1) == 0) {
            writeLAMMPSTRJ(filename, true, running_time);
        }  
        
        running_time+=timestep;       
        
    }
    return 1;
}
