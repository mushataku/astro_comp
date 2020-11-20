/*****************************************
[補足]
*****************************************/

#define _USE_MATH_DEFINES
#include <stdlib.h>
#include <stdio.h>
#include <cmath>
#include <iostream>
#include <vector>
#include <time.h>
#include <algorithm>
#include <random>
#include <sys/stat.h>
#include <utility> // pair 

using vd = std::vector<double>;
using vvd = std::vector<vd>;
using vi = std::vector<int>;
using vvi = std::vector<vi>;


/**********************config*************************/
// 0:熱浴法 1:メトロポリス法
const int METHOD = 1;
// 0:ランダム 1:格子状 2:網状 3:基底状態
const int INIT = 0;
// 0:細かく 1:十点だけ取る
const int OUT_T = 1;
// 0:熱平衡は全てサンプリング 1:配位間を空けてサンプリング 2:その2
const int DISTANCE = 3;
/*****************************************************/

/******************計算条件********************/
// 1次元方向のスピンの数
const int n = 100;
const double Nspin = double(n)*n;
//エネルギーのサンプルの数
const int Nsample = 1e4;
// 熱平衡状態までの回数
const int TE_ITE = 1e3;
// 温度T:k_bT/J は [0.1,5]
// const double T_ini = 2.0;
const double T_ini = 0.1;
const double T_fin = 5.0;
const double dT1 = 0.5;
const double dT2 = 0.01;
const double dT3 = 0.5;
// const double dT1 = 0.1;
// const double dT2 = 0.2;
// const double dT3 = 0.5;
/*********************************************/


/**********************一様乱数の定義*************************/
//非決定的な乱数生成機、これを使ってシードをランダムに選ぶ
std::random_device rnddev;
//シード値の指定に使う乱数
std::mt19937 mt(rnddev());
//一様乱数
std::uniform_int_distribution<int> rndn(0,n);
std::uniform_int_distribution<int> rnd2(0,1);
std::uniform_real_distribution<double> rndd(0.0,1.0);
/************************************************************/

/*******************************関数の定義*******************************/
//配列qに初期のスピン配位をランダムに入れる
void init_configuration(vvi &q);
void set_Tvector(vd &vT);
// 熱浴法で使う遷移確率の更新
void set_P_p(vd &P_p, double T);
//配位をupdate
void update(vvi &q, double T);
// 温度Tでの熱平衡状態を取得
int get_equilibrium(vvi &q, double T);
// 熱平衡状態から物理量のサンプリングを行う
void sampling(vvi &q, vd &e_list, vd &m_list, double T, int TE_ite);
// eをget
double calc_e(vvi &q, double T);
// mをget
double calc_m(vvi &q);
// 配列の統計量（平均と標準偏差）を計算
std::pair<double,double> mean_deviation(vd &v);
// 統計量から熱力学量を計算
void calc_result(vd &e_list, vd &m_list, double T);
// 計算時間を出力
void calc_time(int t);
/***********************************************************************/
// 完全楕円積分の値を出力
void ellips();

// 配位の更新判定に使う確率のテーブルを予め計算・保存するクラス
class Probability{
public:
  vd plus;
  vd exp_;
  Probability(){
    plus.resize(5);
    exp_.resize(5);
  }
  void set(double T);
};
Probability P;

// 計算データを保存するディレクトリを生成・保存するクラス
class my_directories{
public:
  char DATA_PATH[100];
  char result_filename[100];
  char DIST_PATH[100];
  char TE_PATH[100];
  char CONF_TE_PATH[100];
  my_directories(){
    init_dir();
  }
private:
  void init_dir();
};
my_directories my_dir;

int main(){
  // ellips();
  clock_t start_t, end_t;
  start_t = time(NULL);
  
  vd vT;//計算する温度を入れる配列
  // 全てのスピンの状態を格納 q = {1,-1,1,-1,-1,...}
  vvi q(n, vi(n));
  vd e_list(Nsample);
  vd m_list(Nsample);
  int TE_ite;
  
  //初期化
  set_Tvector(vT);
  init_configuration(q);

  /*******************記録用のフォルダ、ファイル作成*******************/
  
  /*****************************************************************/

  printf("spin number:%d\n", n);
  //各温度について比熱を計算

  printf("****************CALICULATION CONDITIONS****************\n");
  for(auto T : vT){
    printf("temperature:%f\n", T);
    P.set(T);
    // 温度Tでの熱平衡状態を取得
    TE_ite = get_equilibrium(q, T);
    // 熱平衡状態から配位をサンプリングする
    sampling(q, e_list, m_list, T, TE_ite);
    // サンプルからエネルギー・比熱・磁化を計算・出力
    calc_result(e_list, m_list, T);
  }
  printf("******************NORMAL END******************\n");
  end_t = time(NULL);
  calc_time((int)(end_t-start_t));
  return 0;
}

////////function/////////////

double calc_e(vvi &q, double T){
  double energy = 0.0;
  for(int i = 0; i < n; i++) for(int j = 0; j < n; j++){
    energy += q[i][j]*( q[i][(j+1)%n] + q[(i+1)%n][j] );
  }
  return -energy/(T*Nspin);
}

double calc_m(vvi &q){
  double M = 0.0;
  for(int i = 0; i < n; i++) for(int j = 0; j < n; j++){
    M += q[i][j];
  }
  return M/Nspin;
}

std::pair<double,double> mean_deviation(vd &v){
  int num = (int)v.size();
  double sum = 0.0, sum2 = 0.0;
  double mean, mean2, deviation;
  for(int i = 0; i < num; i++) {
    sum += v[i];
    sum2 += v[i]*v[i];
  }
  mean = sum/num;
  mean2 = sum2/num;
  deviation = std::sqrt(std::abs(mean2-mean*mean)/num);
  // printf("%e %e %e %e\n", mean2,mean*mean,mean2-mean*mean,deviation);
  return std::make_pair(mean,deviation);
}

void calc_result(vd &e_list, vd &m_list, double T){
  FILE *result_fp = fopen(my_dir.result_filename,"a");
  double e,c,m,e_sig,c_sig,m_sig;
  std::pair<double,double> e_mean_deviation, m_mean_deviation;

  e_mean_deviation = mean_deviation(e_list);
  m_mean_deviation = mean_deviation(m_list);

  e = e_mean_deviation.first;
  e_sig = e_mean_deviation.second;
  c = Nspin*(double)Nsample*e_sig*e_sig;
  c_sig = 0.0;
  m = std::abs(m_mean_deviation.first);
  m_sig = m_mean_deviation.second;

  fprintf(result_fp, "%e,%e,%e,%e,%e,%e,%e\n", T,e,c,m,e_sig,c_sig,m_sig);
  printf("e:%e c:%e m:%e\n", e,c,m);
  fclose(result_fp);
}

void sampling(vvi &q, vd &e_list, vd &m_list, double T, int TE_ite){
  char dist_filename[100];
  FILE *dist_fp;
  sprintf(dist_filename, "%s/T%.3f.csv", my_dir.DIST_PATH, T);
  dist_fp = fopen(dist_filename, "w");
  fprintf(dist_fp, "No,Energy,Magnetization,T=,%e\n",T);
  int between;

  if(DISTANCE == 0) between = 1;

  if(DISTANCE == 1){
    if(T < 0.9) between = 1;
    else if(T < 2.9) between = 100;
    else between = 10;
  }
  if(DISTANCE == 2){
    if(T < 0.9) between = 1;
    if(T < 2.1) between = 100;
    else if(T < 2.5) between = 1000;
    else if(T < 2.9) between = 100;
    else between = 10;
  }
  if(DISTANCE == 3){
    if(T < 0.9) between = 1;
    if(T < 2.1) between = 100;
    else if(T < 2.5) between = 10000;
    else if(T < 2.9) between = 100;
    else between = 10;
  }

  // カノニカル分布にしたがったエネルギーをNsample個getする
  for(int i = 0; i < Nsample; i++) {
    //相関を消すために数個置きに配位を選ぶ -> なんか意味ない気がするなあ -> めっちゃ意味あった
    for(int j = 0; j < between; j++) {
      //配位をupdate
      update(q, T);    
    }
    e_list[i] = calc_e(q, T);
    m_list[i] = calc_m(q);
    fprintf(dist_fp, "%d,%e,%e\n", i+TE_ite, e_list[i], m_list[i]);
  }
  fclose(dist_fp);
}

void init_configuration(vvi &q){
  for(int i = 0; i < n; i++) for(int j = 0; j < n; j++) {
    if(INIT == 0){
      //randが偶数なら上向き、奇数なら下向き
      if(rnd2(mt) == 0) q[i][j] = 1;
      else q[i][j] = -1;
    }
    // 格子状
    if(INIT == 1){
      if((i+j)%2 == 0) q[i][j] = 1;
      else q[i][j] = -1;
    }
    // あみあみ
    if(INIT == 2){
      if(i%2 == 0) q[i][j] = 1;
      else q[i][j] = -1;
    }
    // 基底状態
    if(INIT == 3){
      q[i][j] = 1;
    }
  }
}

void set_Tvector(vd &vT){
  if(OUT_T == 0){
    //0.1 ～ 1.0 はdT1刻み 1.0 ~ 2.0 はdT2刻み それ以降はdT2刻み
    double T = T_ini, dT = dT1;
    while(T < T_fin + 1e-9){
      vT.push_back(T);
      if(T > 2.0 - 1e-9) dT = dT2;
      if(T > 2.5 - 1e-9) dT = dT3;
      T += dT;
    }
  }

  if(OUT_T == 1){
    // 手で決める
    vT = {0.1,1.0,2.0,2.26,2.3,2.4,2.6,3.0,4.0,5.0};
    std::reverse(vT.begin(),vT.end());
  }

  // char T_filename[100];
  // FILE *T_fp;
  // sprintf(T_filename, "%s/tau.csv", my_dir.DIST_PATH, n, T);
  // T_fp = fopen(T_filename, "w");
  // fprintf(T_fp, "tau,Energy,Magnetization,T=,%e\n",T);
}

void update(vvi &q, double T){
  int m, i_p, i_m, j_p, j_m;
  // 熱浴法
  if(METHOD == 0){
    // vvi nq(n,vi(n));
    for(int i = 0; i < n; i++) for(int j = 0; j < n; j++) {
      i_m = i-1, i_p = i+1, j_p = j+1, j_m = j-1;
      if(i_p == n) i_p = 0;
      if(i_m == -1) i_m = n-1;
      if(j_p == n) j_p = 0;
      if(j_m == -1) j_m = n-1;

      m = q[i][j_p] + q[i][j_m] + q[i_p][j] + q[i_m][j];
      if(rndd(mt) < P.plus[(m+4)/2]) q[i][j] = 1;
      else q[i][j] = -1;

      // if(rndd(mt) < P.plus[(m+4)/2]) nq[i][j] = 1;
      // else nq[i][j] = -1;
    }
    // q = nq;
    return;
  }

  // メトロポリス法
  int trial;
  for(int i = 0; i < n; i++) for(int j = 0; j < n; j++) {
    trial = (rndd(mt) < 0.5 ? 1 : -1);
    if(q[i][j] == trial) continue;

    i_m = i-1, i_p = i+1, j_p = j+1, j_m = j-1;
    if(i_p == n) i_p = 0;
    if(i_m == -1) i_m = n-1;
    if(j_p == n) j_p = 0;
    if(j_m == -1) j_m = n-1;
    m = (q[i][j_p] + q[i][j_m] + q[i_p][j] + q[i_m][j])*(trial-q[i][j]);
    // printf("m:%d P:%e\n", m, P.exp_[(m+8)/4]);
    if(rndd(mt) < P.exp_[(m+8)/4]){
      q[i][j] = trial;
      // printf("accept!!\n");
    }
  }
}

int get_equilibrium(vvi &q, double T){
  char TE_filename[100], conf_TE_filename[100];
  FILE *TE_fp, *conf_TE_fp;
  double e,m;
  int ite = 1;

  sprintf(TE_filename, "%s/T%.3f.csv", my_dir.TE_PATH, T);
  sprintf(conf_TE_filename, "%s/T%.3f.csv", my_dir.CONF_TE_PATH, T);
  TE_fp = fopen(TE_filename, "w");
  conf_TE_fp = fopen(conf_TE_filename, "w");
  fprintf(TE_fp, "No,Energy,Magnetization,T=,%e\n",T);
  
  for(ite = 1; ite <= TE_ITE; ite++){
    update(q, T);
    e = calc_e(q, T);
    m = calc_m(q);
    fprintf(TE_fp, "%d,%e,%e\n",ite,e,m);
  }

  fprintf(conf_TE_fp, "i,j,q,T=,%e\n",T);
  for(int i = 0; i < n; i++) {
    for(int j = 0; j < n; j++) {
      fprintf(conf_TE_fp, "%d,%d,%d\n",i,j,q[i][j]);
    }
  }
  fclose(TE_fp);
  fclose(conf_TE_fp);
  // exit(-1);
  return ite;
}

void calc_time(int t){
  int day,hour,minute,second;
  day = t/(60*60*24);
  t = t%(60*60*24);
  hour = t/(60*60);
  t = t%(60*60);
  minute = t/60;
  t = t%60;
  second = t;
  printf("This calculatioin took %d day %d hour %d minute %d second\n\n", day, hour, minute, second);
}

void Probability::set(double T){
  double tmp;
  vd m = {-4.0,-2.0,0.0,2.0,4.0};
  for(int i = 0; i < 5; i++) {
    tmp = std::exp(m[i]/T) + std::exp(-m[i]/T);
    plus[i] = std::exp(m[i]/T)/tmp;
    exp_[i] = std::exp(2.0*m[i]/T);
  }
}

void my_directories::init_dir(){
  mkdir("./data", 0777);
  if(METHOD == 0) sprintf(DATA_PATH, "./data/heat_bath");
  if(METHOD == 1) sprintf(DATA_PATH, "./data/metropolis");
  mkdir(DATA_PATH, 0777);
  if(DISTANCE == 0){
    sprintf(DATA_PATH, "%s/spin%dsample%.1e", DATA_PATH, n, (double)Nsample);
  }
  if(DISTANCE != 0){
    sprintf(DATA_PATH, "%s/spin%dsample%.1edistance%d", DATA_PATH, n, (double)Nsample, DISTANCE);
    printf("%s\n", DATA_PATH);
  }
  mkdir(DATA_PATH, 0777);
  sprintf(DIST_PATH, "%s/distribution", DATA_PATH);
  mkdir(DIST_PATH, 0777);
  sprintf(TE_PATH, "%s/TE", DATA_PATH);
  mkdir(TE_PATH, 0777);
  sprintf(CONF_TE_PATH, "%s/conf_TE", DATA_PATH);
  mkdir(CONF_TE_PATH, 0777);

  sprintf(result_filename, "%s/result.csv", DATA_PATH);
  FILE *result_fp = fopen(result_filename,"w");
  fprintf(result_fp, "T,e,c,m,e_sig,c_sig,m_sig\n");
  fclose(result_fp);
}

void ellips(){
  FILE *ellips_fp = fopen("./data/ellips.csv","w");
  fprintf(ellips_fp, "k,K,E\n");
  int NUM = 100;
  double k,K,E;
  for(int i = 0; i < NUM; i++) {
    k = (double)i/NUM;
    K = std::comp_ellint_1(k);
    E = std::comp_ellint_2(k);
    fprintf(ellips_fp, "%e,%e,%e\n", k,K,E);
  }
}