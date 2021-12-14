
#ifndef SSLWE_CONFIG_SET
#include <iostream>
#include <cstdint>
#include <vector>
#include <array>

#include "m4ri/m4ri.h"
#include "test/mceliece/challenges/mce640.h"
constexpr uint32_t G_w = w;
constexpr uint32_t G_l =2;
constexpr uint32_t G_l1 =2;
constexpr uint32_t G_p =1;
constexpr uint32_t G_epsilon =0;
constexpr uint64_t CUTOFF=0;
constexpr uint64_t CUTOFF_RETRIES=0ul;
constexpr uint64_t r1=0;
constexpr uint32_t HM1_NRB=2;
constexpr uint32_t HM2_NRB=11;
constexpr uint32_t HM1_SIZEB=80;
constexpr uint32_t HM2_SIZEB=20;
#define NUMBER_THREADS 2
#define NUMBER_OUTER_THREADS 1
#define USE_DOOM 0
#define USE_MO 1
#define USE_NN 0
#define BJMM_DOOM_SPECIAL_FORM 0
#define PRINT_LOOPS 1000
#define FULLLENGTH 0
#define LOW_WEIGHT 0
#define SYNDROM 0
#define TERNARY 0
constexpr uint32_t MO_NRHM=1;
constexpr uint32_t MO_l2=11;
constexpr uint32_t HM1_USESTDBINARYSEARCH=true;
constexpr uint32_t HM2_USESTDBINARYSEARCH=true;
constexpr uint32_t HM1_USEINTERPOLATIONSEARCH=false;
constexpr uint32_t HM2_USEINTERPOLATIONSEARCH=false;
constexpr uint32_t HM1_USELINEARSEARCH=false;
constexpr uint32_t HM2_USELINEARSEARCH=false;
constexpr uint32_t HM1_USELOAD=true;
constexpr uint32_t HM2_USELOAD=true;
constexpr uint32_t TERNARY_NR1=-1;
constexpr uint32_t TERNARY_NR2=-1;
constexpr uint32_t TERNARY_ALPHA=-1;
constexpr uint32_t TERNARY_FILTER2=1;
constexpr uint32_t TERNARY_ENUMERATION_TYPE=-1;

#include "helper.h"
#include "matrix.h"
#include "bjmm.h"
#include "mo.h"
#include "ternary.h"
#endif //SSLWE_CONFIG_SET