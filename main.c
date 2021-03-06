/*
--------------------------------------------------
    91f7d09794d8da29f028e77df49d4907
    https://github.com/DaisyGAN/
--------------------------------------------------
    DaisyGANv5

    Technically not a generative adversarial network anymore.

    rndBest() & bestSetting() allows a multi-process model
*/

#pragma GCC diagnostic ignored "-Wunused-result"
#pragma GCC diagnostic ignored "-Wformat-zero-length"

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <fcntl.h>
#include <locale.h>
#include <sys/file.h>

#define uint uint32_t
#define NO_LEARN -2

///

#define TABLE_SIZE_MAX 160000
#define DIGEST_SIZE 8
#define WORD_SIZE 256 //32
#define MESSAGE_SIZE WORD_SIZE*DIGEST_SIZE

///

// #define FAST_PREDICTABLE_MODE
// #define DATA_TRAIN_PERCENT 0.7
// #define DATA_SIZE 3045 //110927
// #define OUTPUT_QUOTES 33333
// #define FIRSTLAYER_SIZE 64
// #define HIDDEN_SIZE 128
// #define TRAINING_LOOPS 1
// uint        _linit      = 1;
// float       _lrate      = 0.016559;
// float       _ldecay     = 0.0005;
// float       _ldropout   = 0.130533;
// uint        _lbatches   = 1;
// uint        _loptimiser = 1;
// float       _lmomentum  = 0.530182;
// float       _lrmsalpha  = 0.578107;
// const float _lgain      = 1.0;

// #define FAST_PREDICTABLE_MODE
// #define DATA_TRAIN_PERCENT 0.7
// #define DATA_SIZE 3045 //110927
// #define OUTPUT_QUOTES 33333
// #define FIRSTLAYER_SIZE 128
// #define HIDDEN_SIZE 256
// #define TRAINING_LOOPS 1
// uint        _linit      = 1;
// float       _lrate      = 0.016559;
// float       _ldecay     = 0.0005;
// float       _ldropout   = 0.130533;
// uint        _lbatches   = 8;
// uint        _loptimiser = 1;
// float       _lmomentum  = 0.530182;
// float       _lrmsalpha  = 0.578107;
// const float _lgain      = 1.0;

// this is not the vegetarian option
#define FAST_PREDICTABLE_MODE
#define DATA_TRAIN_PERCENT 0.7
#define DATA_SIZE 3045 //110927
#define OUTPUT_QUOTES 33333
#define FIRSTLAYER_SIZE 512
#define HIDDEN_SIZE 1024
#define TRAINING_LOOPS 1
uint        _linit      = 3;
float       _lrate      = 0.016325;
float       _ldecay     = 0.0005;
float       _ldropout   = 0.130533;
uint        _lbatches   = 16;
uint        _loptimiser = 1;
float       _lmomentum  = 0.530182;
float       _lrmsalpha  = 0.578107;
const float _lgain      = 1.0;

//

uint _log = 0;

struct
{
    float* data;
    float* momentum;
    float bias;
    float bias_momentum;
    uint weights;
}
typedef ptron;

// discriminator 
ptron d1[FIRSTLAYER_SIZE];
ptron d2[HIDDEN_SIZE];
ptron d3[HIDDEN_SIZE];
ptron d4;

// normalised training data
float digest[DATA_SIZE][DIGEST_SIZE] = {0};

//word lookup table / index
char wtable[TABLE_SIZE_MAX][WORD_SIZE] = {0};
uint TABLE_SIZE = 0;
uint TABLE_SIZE_H = 0;


//*************************************
// utility functions
//*************************************

void loadTable(const char* file)
{
    FILE* f = fopen(file, "r");
    if(f)
    {
        uint index = 0;
        while(fgets(wtable[index], WORD_SIZE, f) != NULL)
        {
            char* pos = strchr(wtable[index], '\n');
            if(pos != NULL)
                *pos = '\0';
            
            index++;
            if(index == TABLE_SIZE_MAX)
                break;
        }
        TABLE_SIZE = index;
        TABLE_SIZE_H = TABLE_SIZE / 2;
        fclose(f);
    }
}

float getWordNorm(const char* word)
{
    for(uint i = 0; i < TABLE_SIZE; i++)
        if(strcmp(word, wtable[i]) == 0)
            return (((double)i) / (double)(TABLE_SIZE_H))-1.0;

    return 0;
}

void saveWeights()
{
    FILE* f = fopen("weights.dat", "w");
    if(f != NULL)
    {
        if(flock(fileno(f), LOCK_EX) == -1)
            printf("ERROR flock(LOCK_EX) in saveWeights()\n");

        for(uint i = 0; i < FIRSTLAYER_SIZE; i++)
        {
            if(fwrite(&d1[i].data[0], 1, d1[i].weights*sizeof(float), f) != d1[i].weights*sizeof(float))
                printf("ERROR fwrite() in saveWeights() #1w\n");
            
            if(fwrite(&d1[i].momentum[0], 1, d1[i].weights*sizeof(float), f) != d1[i].weights*sizeof(float))
                printf("ERROR fwrite() in saveWeights() #1m\n");

            if(fwrite(&d1[i].bias, 1, sizeof(float), f) != sizeof(float))
                printf("ERROR fwrite() in saveWeights() #1w\n");
            
            if(fwrite(&d1[i].bias_momentum, 1, sizeof(float), f) != sizeof(float))
                printf("ERROR fwrite() in saveWeights() #1m\n");
        }

        for(uint i = 0; i < HIDDEN_SIZE; i++)
        {
            if(fwrite(&d2[i].data[0], 1, d2[i].weights*sizeof(float), f) != d2[i].weights*sizeof(float))
                printf("ERROR fwrite() in saveWeights() #1w\n");
            
            if(fwrite(&d2[i].momentum[0], 1, d2[i].weights*sizeof(float), f) != d2[i].weights*sizeof(float))
                printf("ERROR fwrite() in saveWeights() #1m\n");

            if(fwrite(&d2[i].bias, 1, sizeof(float), f) != sizeof(float))
                printf("ERROR fwrite() in saveWeights() #1w\n");
            
            if(fwrite(&d2[i].bias_momentum, 1, sizeof(float), f) != sizeof(float))
                printf("ERROR fwrite() in saveWeights() #1m\n");
        }

        for(uint i = 0; i < HIDDEN_SIZE; i++)
        {
            if(fwrite(&d3[i].data[0], 1, d3[i].weights*sizeof(float), f) != d3[i].weights*sizeof(float))
                printf("ERROR fwrite() in saveWeights() #1w\n");
            
            if(fwrite(&d3[i].momentum[0], 1, d3[i].weights*sizeof(float), f) != d3[i].weights*sizeof(float))
                printf("ERROR fwrite() in saveWeights() #1m\n");

            if(fwrite(&d3[i].bias, 1, sizeof(float), f) != sizeof(float))
                printf("ERROR fwrite() in saveWeights() #1w\n");
            
            if(fwrite(&d3[i].bias_momentum, 1, sizeof(float), f) != sizeof(float))
                printf("ERROR fwrite() in saveWeights() #1m\n");
        }

        if(fwrite(&d4.data[0], 1, d4.weights*sizeof(float), f) != d4.weights*sizeof(float))
            printf("ERROR fwrite() in saveWeights() #1w\n");
        
        if(fwrite(&d4.momentum[0], 1, d4.weights*sizeof(float), f) != d4.weights*sizeof(float))
            printf("ERROR fwrite() in saveWeights() #1m\n");

        if(fwrite(&d4.bias, 1, sizeof(float), f) != sizeof(float))
            printf("ERROR fwrite() in saveWeights() #1w\n");
        
        if(fwrite(&d4.bias_momentum, 1, sizeof(float), f) != sizeof(float))
            printf("ERROR fwrite() in saveWeights() #1m\n");

        if(flock(fileno(f), LOCK_UN) == -1)
            printf("ERROR flock(LOCK_UN) in saveWeights()\n");

        fclose(f);
    }
}

void loadWeights()
{
    FILE* f = fopen("weights.dat", "r");
    if(f == NULL)
    {
        printf("!!! no pre-existing weights where found, starting from random initialisation.\n\n\n-----------------\n");
        return;
    }

    for(uint i = 0; i < FIRSTLAYER_SIZE; i++)
    {
        while(fread(&d1[i].data[0], 1, d1[i].weights*sizeof(float), f) != d1[i].weights*sizeof(float))
            sleep(333);

        while(fread(&d1[i].momentum[0], 1, d1[i].weights*sizeof(float), f) != d1[i].weights*sizeof(float))
            sleep(333);

        while(fread(&d1[i].bias, 1, sizeof(float), f) != sizeof(float))
            sleep(333);

        while(fread(&d1[i].bias_momentum, 1, sizeof(float), f) != sizeof(float))
            sleep(333);
    }

    for(uint i = 0; i < HIDDEN_SIZE; i++)
    {
        while(fread(&d2[i].data[0], 1, d2[i].weights*sizeof(float), f) != d2[i].weights*sizeof(float))
            sleep(333);

        while(fread(&d2[i].momentum[0], 1, d2[i].weights*sizeof(float), f) != d2[i].weights*sizeof(float))
            sleep(333);

        while(fread(&d2[i].bias, 1, sizeof(float), f) != sizeof(float))
            sleep(333);

        while(fread(&d2[i].bias_momentum, 1, sizeof(float), f) != sizeof(float))
            sleep(333);
    }

    for(uint i = 0; i < HIDDEN_SIZE; i++)
    {
        while(fread(&d3[i].data[0], 1, d3[i].weights*sizeof(float), f) != d3[i].weights*sizeof(float))
            sleep(333);

        while(fread(&d3[i].momentum[0], 1, d3[i].weights*sizeof(float), f) != d3[i].weights*sizeof(float))
            sleep(333);

        while(fread(&d3[i].bias, 1, sizeof(float), f) != sizeof(float))
            sleep(333);

        while(fread(&d3[i].bias_momentum, 1, sizeof(float), f) != sizeof(float))
            sleep(333);
    }

    while(fread(&d4.data[0], 1, d4.weights*sizeof(float), f) != d4.weights*sizeof(float))
            sleep(333);

    while(fread(&d4.momentum[0], 1, d4.weights*sizeof(float), f) != d4.weights*sizeof(float))
            sleep(333);

    while(fread(&d4.bias, 1, sizeof(float), f) != sizeof(float))
            sleep(333);

    while(fread(&d4.bias_momentum, 1, sizeof(float), f) != sizeof(float))
        sleep(333);

    fclose(f);
}

float qRandFloat(const float min, const float max)
{
#ifndef FAST_PREDICTABLE_MODE
    static time_t ls = 0;
    if(time(0) > ls)
    {
        srand(time(0));
        ls = time(0) + 33;
    }
#endif
    const float rv = (float)rand();
    if(rv == 0)
        return min;
    return ( (rv / (float)RAND_MAX) * (max-min) ) + min;
}

float uRandFloat(const float min, const float max)
{
#ifdef FAST_PREDICTABLE_MODE
    return qRandFloat(min, max);
#else
    int f = open("/dev/urandom", O_RDONLY | O_CLOEXEC);
    uint s = 0;
    ssize_t result = read(f, &s, 4);
    srand(s);
    close(f);
    const float rv = (float)rand();
    if(rv == 0)
        return min;
    return ( (rv / (float)RAND_MAX) * (max-min) ) + min;
#endif
}

float qRandWeight(const float min, const float max)
{
#ifndef FAST_PREDICTABLE_MODE
    static time_t ls = 0;
    if(time(0) > ls)
    {
        srand(time(0));
        ls = time(0) + 33;
    }
#endif
    float pr = 0;
    while(pr == 0) //never return 0
    {
        const float rv = (float)rand();
        if(rv == 0)
            return min;
        const float rv2 = ( (rv / (float)RAND_MAX) * (max-min) ) + min;
        pr = roundf(rv2 * 100) / 100; // two decimals of precision
    }
    return pr;
}

float uRandWeight(const float min, const float max)
{
#ifdef FAST_PREDICTABLE_MODE
    return qRandWeight(min, max);
#else
    int f = open("/dev/urandom", O_RDONLY | O_CLOEXEC);
    uint s = 0;
    ssize_t result = read(f, &s, 4);
    srand(s);
    close(f);
    float pr = 0;
    while(pr == 0) //never return 0
    {
        const float rv = (float)rand();
        if(rv == 0)
            return min;
        const float rv2 = ( (rv / (float)RAND_MAX) * (max-min) ) + min;
        pr = roundf(rv2 * 100) / 100; // two decimals of precision
    }
    return pr;
#endif
}

uint qRand(const uint min, const uint umax)
{
#ifndef FAST_PREDICTABLE_MODE
    static time_t ls = 0;
    if(time(0) > ls)
    {
        srand(time(0));
        ls = time(0) + 33;
    }
#endif
    const int rv = rand();
    const uint max = umax + 1;
    if(rv == 0)
        return min;
    return ( ((float)rv / (float)RAND_MAX) * (max-min) ) + min; //(rand()%(max-min))+min;
}

uint uRand(const uint min, const uint umax)
{
#ifdef FAST_PREDICTABLE_MODE
    return qRand(min, umax);
#else
    int f = open("/dev/urandom", O_RDONLY | O_CLOEXEC);
    uint s = 0;
    ssize_t result = read(f, &s, 4);
    srand(s);
    close(f);
    const int rv = rand();
    const uint max = umax + 1;
    if(rv == 0)
        return min;
    return ( ((float)rv / (float)RAND_MAX) * (max-min) ) + min; //(rand()%(max-min))+min;
#endif
}

void newSRAND()
{
    struct timespec c;
    clock_gettime(CLOCK_MONOTONIC, &c);
    srand(time(0)+c.tv_nsec);
}

//https://stackoverflow.com/questions/30432856/best-way-to-get-number-of-lines-in-a-file-c
uint countLines(const char* file)
{
    uint lines = 0;
    FILE *fp = fopen(file, "r");
    if(fp != NULL)
    {
        while(EOF != (fscanf(fp, "%*[^\n]"), fscanf(fp,"%*c")))
            ++lines;
        
        fclose(fp);
    }
    return lines;
}

void clearFile(const char* file)
{
    FILE *f = fopen(file, "w");
    if(f != NULL)
    {
        fprintf(f, "");
        fclose(f);
    }
}

void timestamp()
{
    const time_t ltime = time(0);
    printf("%s", asctime(localtime(&ltime)));
}


//*************************************
// create layer
//*************************************

void createPerceptron(ptron* p, const uint weights, const float d)
{
    p->data = malloc(weights * sizeof(float));
    if(p->data == NULL)
    {
        printf("Perceptron creation failed (w)%u.\n", weights);
        return;
    }

    p->momentum = malloc(weights * sizeof(float));
    if(p->momentum == NULL)
    {
        printf("Perceptron creation failed (m)%u.\n", weights);
        return;
    }

    p->weights = weights;

    //const float d = 1/sqrt(p->weights);
    for(uint i = 0; i < p->weights; i++)
    {
        p->data[i] = qRandWeight(-d, d); //qRandWeight(-1, 1);
        p->momentum[i] = 0;
    }

    p->bias = 0; //qRandWeight(-1, 1);
    p->bias_momentum = 0;
}

void resetPerceptron(ptron* p, const float d)
{
    //const float d = 1/sqrt(p->weights);
    for(uint i = 0; i < p->weights; i++)
    {
        p->data[i] = qRandWeight(-d, d); //qRandWeight(-1, 1);
        p->momentum[i] = 0;
    }

    p->bias = 0; //qRandWeight(-1, 1);
    p->bias_momentum = 0;
}

void createPerceptrons()
{
    const uint init_method = _linit;
    float l1d = 1;
    float l2d = 1;
    float l3d = 1;
    float l4d = 1;

    // Xavier uniform
    if(init_method == 1)
    {
        l1d = sqrt(6.0/(FIRSTLAYER_SIZE+HIDDEN_SIZE));
        l2d = sqrt(6.0/(HIDDEN_SIZE+HIDDEN_SIZE));
        l3d = sqrt(6.0/(HIDDEN_SIZE+HIDDEN_SIZE));
        l4d = sqrt(6.0/(HIDDEN_SIZE+1));
    }

    // LeCun uniform
    if(init_method == 2)
    {
        l1d = sqrt(3.0/DIGEST_SIZE);
        l2d = sqrt(3.0/FIRSTLAYER_SIZE);
        l3d = sqrt(3.0/HIDDEN_SIZE);
        l4d = sqrt(3.0/HIDDEN_SIZE);
    }

    // What I thought was LeCun
    if(init_method == 3)
    {
        l1d = pow(DIGEST_SIZE, 0.5);
        l2d = pow(FIRSTLAYER_SIZE, 0.5);
        l3d = pow(HIDDEN_SIZE, 0.5);
        l4d = pow(HIDDEN_SIZE, 0.5);
    }
    
    //printf("%f %f %f %f \n", l1d, l2d, l3d, l4d);

    for(int i = 0; i < FIRSTLAYER_SIZE; i++)
        createPerceptron(&d1[i], DIGEST_SIZE, l1d);
    for(int i = 0; i < HIDDEN_SIZE; i++)
        createPerceptron(&d2[i], FIRSTLAYER_SIZE, l2d);
    for(int i = 0; i < HIDDEN_SIZE; i++)
        createPerceptron(&d3[i], HIDDEN_SIZE, l3d);
    createPerceptron(&d4, HIDDEN_SIZE, l4d);
}

void resetPerceptrons()
{
    const uint init_method = _linit;
    float l1d = 1;
    float l2d = 1;
    float l3d = 1;
    float l4d = 1;

    // Xavier uniform
    if(init_method == 1)
    {
        l1d = sqrt(6.0/(FIRSTLAYER_SIZE+HIDDEN_SIZE));
        l2d = sqrt(6.0/(HIDDEN_SIZE+HIDDEN_SIZE));
        l3d = sqrt(6.0/(HIDDEN_SIZE+HIDDEN_SIZE));
        l4d = sqrt(6.0/(HIDDEN_SIZE+1));
    }

    // LeCun uniform
    if(init_method == 2)
    {
        l1d = sqrt(3.0/DIGEST_SIZE);
        l2d = sqrt(3.0/FIRSTLAYER_SIZE);
        l3d = sqrt(3.0/HIDDEN_SIZE);
        l4d = sqrt(3.0/HIDDEN_SIZE);
    }

    // What I thought was LeCun
    if(init_method == 3)
    {
        l1d = pow(DIGEST_SIZE, 0.5);
        l2d = pow(FIRSTLAYER_SIZE, 0.5);
        l3d = pow(HIDDEN_SIZE, 0.5);
        l4d = pow(HIDDEN_SIZE, 0.5);
    }

    //printf("%f %f %f %f \n", l1d, l2d, l3d, l4d);

    for(int i = 0; i < FIRSTLAYER_SIZE; i++)
        resetPerceptron(&d1[i], l1d);
    for(int i = 0; i < HIDDEN_SIZE; i++)
        resetPerceptron(&d2[i], l2d);
    for(int i = 0; i < HIDDEN_SIZE; i++)
        resetPerceptron(&d3[i], l3d);
    resetPerceptron(&d4, l4d);
}


//*************************************
// activation functions
// https://en.wikipedia.org/wiki/Activation_function
// https://www.analyticsvidhya.com/blog/2020/01/fundamentals-deep-learning-activation-functions-when-to-use-them/
// https://adl1995.github.io/an-overview-of-activation-functions-used-in-neural-networks.html
// https://stackoverflow.com/questions/42537957/fast-accurate-atan-arctan-approximation-algorithm
// https://varietyofsound.wordpress.com/2011/02/14/efficient-tanh-computation-using-lamberts-continued-fraction/
//*************************************

float fast_tanh(float x)
{
    float x2 = x * x;
    float a = x * (135135.0f + x2 * (17325.0f + x2 * (378.0f + x2)));
    float b = 135135.0f + x2 * (62370.0f + x2 * (3150.0f + x2 * 28.0f));
    return a / b;
}

float tanh_c3(float v)
{
    const float c1 = 0.03138777F;
    const float c2 = 0.276281267F;
    const float c_log2f = 1.442695022F;
    v *= c_log2f;
    int intPart = (int)v;
    float x = (v - intPart);
    float xx = x * x;
    float v1 = c_log2f + c2 * xx;
    float v2 = x + xx * c1 * x;
    float v3 = (v2 + v1);
    *((int*)&v3) += intPart << 24;
    float v4 = v2 - v1;
    return (v3 + v4) / (v3 - v4);
}

float fatan(float x)
{
    return 0.78539816339*x - x*(fabs(x) - 1)*(0.2447 + 0.0663*fabs(x));
}

float fatan2(float x)
{
    float xx = x * x;
    return ((0.0776509570923569*xx + -0.287434475393028)*xx + 0.995181682)*x;
}

static inline float bipolarSigmoid(float x)
{
    return (1 - exp(-x)) / (1 + exp(-x));
}

static inline float fbiSigmoid(float x)
{
    return (1 - fabs(x)) / (1 + fabs(x));
}

static inline float arctan(float x)
{
    return atan(x);
}

//https://stats.stackexchange.com/questions/60166/how-to-use-1-7159-tanh2-3-x-as-activation-function
static inline float lecun_tanh(float x)
{
    return 1.7159 * tanh(0.666666667 * x);
}

static inline float sigmoid(float x)
{
    return 1 / (1 + exp(-x));
}

static inline float fSigmoid(float x)
{
    return x / (1 + fabs(x));
    //return 0.5 * (x / (1 + abs(x))) + 0.5;
}

static inline float swish(float x)
{
    return x * sigmoid(x);
}

static inline float leakyReLU(float x)
{
    if(x < 0){x *= 0.01;}
    return x;
}

static inline float ReLU(float x)
{
    if(x < 0){x = 0;}
    return x;
}

static inline float ReLU6(float x)
{
    if(x < 0){x = 0;}
    if(x > 6){x = 6;}
    return x;
}

static inline float leakyReLU6(float x)
{
    if(x < 0){x *= 0.01;}
    if(x > 6){x = 6;}
    return x;
}

static inline float smoothReLU(float x) //aka softplus
{
    return log(1 + exp(x));
}

static inline float logit(float x)
{
    return log(x / (1 - x));
}

static inline float bipolar_sigmoidDerivative(float x)
{
    if(x > 0)
        return x * (1 - x);
    else
        return x * (1 + x);
}

static inline float arctanDerivative(float x)
{
    return 1 / (1 + pow(x, 2));
}

static inline float sigmoidDerivative(float x)
{
    return x * (1 - x);
}

static inline float tanhDerivative(float x)
{
    return 1 - pow(x, 2);
}

static inline float ftanhDerivative(float x)
{
    return 1-(x*x);
}

static inline float lecun_tanhDerivative(float x)
{
    //return 1.14393 * pow((1 / cosh(2*x/3)), 2);
    //return 1.14393 * pow((1 / cosh(x * 0.666666666)), 2);
    const float sx = x * 0.6441272;
    return 1.221595 - (sx*sx);
}

static inline float flecun_tanhDerivative(float x)
{
    const float sx = x*0.582784545;
    return 1-(sx*sx);
}

static inline float f2lecun_tanhDerivative(float x)
{
    const float sx = x*0.582784545;
    return 1 - pow(sx, 2);
}

static inline float decay(const float x, const float lambda)
{
    return (1-lambda)*x;
}

static inline float decayL2(const float x, const float lambda)
{
    return (1-_lrate*lambda)*x;
}

void softmax_transform(float* w, const uint32_t n)
{
    float d = 0;
    for(size_t i = 0; i < n; i++)
        d += exp(w[i]);

    for(size_t i = 0; i < n; i++)
        w[i] = exp(w[i]) / d;
}

float crossEntropy(const float predicted, const float expected) //log loss
{
    if(expected == 1)
      return -log(predicted);
    else
      return -log(1 - predicted);
}

float doPerceptron(const float* in, ptron* p)
{
    float ro = 0;
    for(uint i = 0; i < p->weights; i++)
        ro += in[i] * p->data[i];
    ro += p->bias;

    return ro;
}

static inline float SGD(const float input, const float error)
{
    return _lrate * error * input;
}

float Momentum(const float input, const float error, float* momentum)
{
    // const float err = (_lrate * error * input);
    // const float ret = err + _lmomentum * momentum[0];
    // momentum[0] = err;
    // return ret;

    const float err = (_lrate * error * input) + _lmomentum * momentum[0];
    momentum[0] = err;
    return err;
}

float Nesterov(const float input, const float error, float* momentum)
{
    const float vp = momentum[0];
    const float v = _lmomentum * vp + ( _lrate * error * input );
    const float n = v + _lmomentum * (v - momentum[0]);
    momentum[0] = v;
    return n;
}

float ADAGrad(const float input, const float error, float* momentum)
{
    const float err = error * input;
    momentum[0] += err * err;
    return (_lrate / sqrt(momentum[0] + 1e-8)) * err; // 0.00000001
}

float RMSProp(const float input, const float error, float* momentum)
{
    const float err = error * input;
    momentum[0] = _lrmsalpha * momentum[0] + (1 - _lrmsalpha) * (err * err);
    return (_lrate / sqrt(momentum[0] + 1e-8)) * err; // 0.00000001
}

float Optional(const float input, const float error, float* momentum)
{
    if(_loptimiser == 1)
        return Momentum(input, error, momentum);
    else if(_loptimiser == 2)
        return Nesterov(input, error, momentum);
    else if(_loptimiser == 3)
        return ADAGrad(input, error, momentum);
    else if(_loptimiser == 4)
        return RMSProp(input, error, momentum);
    
    return SGD(input, error);
}


//*************************************
// network training functions
//*************************************
float o1[FIRSTLAYER_SIZE] = {0};
float o2[HIDDEN_SIZE] = {0};
float o3[HIDDEN_SIZE] = {0};
float o4 = 0;
float error = 0;
uint batches = 0;

float doDiscriminator(const float* input, const float eo)
{
/**************************************
    Forward Prop
**************************************/

    // layer one, inputs (fc)
    float o1f[FIRSTLAYER_SIZE];
    for(int i = 0; i < FIRSTLAYER_SIZE; i++)
        o1f[i] = lecun_tanh(doPerceptron(input, &d1[i]));

    // layer two, hidden (fc expansion)
    float o2f[HIDDEN_SIZE];
    for(int i = 0; i < HIDDEN_SIZE; i++)
        o2f[i] = lecun_tanh(doPerceptron(&o1f[0], &d2[i]));

    // layer three, hidden (fc)
    float o3f[HIDDEN_SIZE];
    for(int i = 0; i < HIDDEN_SIZE; i++)
        o3f[i] = lecun_tanh(doPerceptron(&o2f[0], &d3[i]));

    // layer four, output (fc compression)
    const float output = sigmoid(lecun_tanh(doPerceptron(&o3f[0], &d4)));

    // if it's just forward pass, return result.
    if(eo == NO_LEARN)
        return output;

/**************************************
    Backward Prop Error
**************************************/

    // reset accumulators if batches was reset
    if(batches == 0)
    {
        memset(&o1, 0x00, FIRSTLAYER_SIZE * sizeof(float));
        memset(&o2, 0x00, HIDDEN_SIZE * sizeof(float));
        memset(&o3, 0x00, HIDDEN_SIZE * sizeof(float));
        o4 = 0;
        error = 0;
    }

    // batch accumulation of outputs
    o4 += output;
    for(int i = 0; i < FIRSTLAYER_SIZE; i++)
        o1[i] += o1f[i];

    for(int i = 0; i < HIDDEN_SIZE; i++)
        o2[i] += o2f[i];

    for(int i = 0; i < HIDDEN_SIZE; i++)
        o3[i] += o3f[i];

    // accumulate output error
    error += eo - output;

    // batching controller
    batches++;
    if(batches < _lbatches)
    {
        return output;
    }
    else
    {
        error /= _lbatches;
        o4 /= _lbatches;
        for(int i = 0; i < FIRSTLAYER_SIZE; i++)
            o1[i] /= _lbatches;
        for(int i = 0; i < HIDDEN_SIZE; i++)
            o2[i] /= _lbatches;
        for(int i = 0; i < HIDDEN_SIZE; i++)
            o3[i] /= _lbatches;
        batches = 0;
    }

    if(error == 0) // superflous will not likely ever happen
        return output;

    float e1[FIRSTLAYER_SIZE];
    float e2[HIDDEN_SIZE];
    float e3[HIDDEN_SIZE];

    float e4 = _lgain * sigmoidDerivative(o4) * error;

    // layer 3 (output)
    float ler = 0;
    for(int j = 0; j < d4.weights; j++)
        ler += d4.data[j] * e4;
    ler += d4.bias * e4;
    
    for(int i = 0; i < HIDDEN_SIZE; i++)
        e3[i] = _lgain * lecun_tanhDerivative(o3[i]) * ler;

    // layer 2
    ler = 0;
    for(int i = 0; i < HIDDEN_SIZE; i++)
    {
        for(int j = 0; j < d3[i].weights; j++)
            ler += d3[i].data[j] * e3[i];
        ler += d3[i].bias * e3[i];
    }
    for(int i = 0; i < HIDDEN_SIZE; i++)
        e2[i] = _lgain * lecun_tanhDerivative(o2[i]) * ler;

    // layer 1
    float k = 0;
    int ki = 0;
    ler = 0;
    for(int i = 0; i < FIRSTLAYER_SIZE; i++)
    {
        for(int j = 0; j < d2[i].weights; j++)
            ler += d2[i].data[j] * e2[i];
        ler += d2[i].bias * e2[i];
    }
    for(int i = 0; i < FIRSTLAYER_SIZE; i++)
    {
        int k0 = 0;
        if(k != 0)
            k0 = 1;
        k += _lgain * lecun_tanhDerivative(o1[i]) * ler;
        if(k0 == 1)
        {
            e1[ki] = k / 2; // i keep forgetting but this hardcoded parameter means the first layer always has to be half the size of the hidden layer
            ki++;
        }
    }

/**************************************
    Update Weights
**************************************/

    // layer 1
    for(int i = 0; i < FIRSTLAYER_SIZE; i++)
    {
        if(_ldropout != 0 && uRandFloat(0, 1) <= _ldropout)
            continue;

        for(int j = 0; j < d1[i].weights; j++)
        {
            // if(_ldecay != 0)
            //     d1[i].data[j] = decayL2(d1[i].data[j], _ldecay);

            d1[i].data[j] += Optional(input[j], e1[i], &d1[i].momentum[j]); //SGD(input[j], e1[i]); //Momentum(input[j], e1[i], &d1[i].momentum[j]);
        }

        d1[i].bias += Optional(1, e1[i], &d1[i].bias_momentum); //SGD(1, e1[i]); //Momentum(1, e1[i], &d1[i].bias_momentum);
    }

    // layer 2
    for(int i = 0; i < HIDDEN_SIZE; i++)
    {
        if(_ldropout != 0 && uRandFloat(0, 1) <= _ldropout)
            continue;

        for(int j = 0; j < d2[i].weights; j++)
        {
            // if(_ldecay != 0)
            //     d2[i].data[j] = decayL2(d2[i].data[j], _ldecay);

            d2[i].data[j] += Optional(o1[j], e2[i], &d2[i].momentum[j]); //SGD(o1[j], e2[i]); //Momentum(o1[j], e2[i], &d2[i].momentum[j]);
        }

        d2[i].bias += Optional(1, e2[i], &d2[i].bias_momentum); //SGD(1, e2[i]); //Momentum(1, e2[i], &d2[i].bias_momentum);
    }

    // layer 3
    for(int i = 0; i < HIDDEN_SIZE; i++)
    {
        if(_ldropout != 0 && uRandFloat(0, 1) <= _ldropout)
            continue;
            
        for(int j = 0; j < d3[i].weights; j++)
        {
            // if(_ldecay != 0)
            //     d3[i].data[j] = decayL2(d3[i].data[j], _ldecay);

            d3[i].data[j] += Optional(o2[j], e3[i], &d3[i].momentum[j]); //SGD(o2[j], e3[i]); //Momentum(o2[j], e3[i], &d3[i].momentum[j]);
        }

        d3[i].bias += Optional(1, e3[i], &d3[i].bias_momentum); //SGD(1, e3[i]); //Momentum(1, e3[i], &d3[i].bias_momentum);
    }

    // layer 4
    for(int j = 0; j < d4.weights; j++)
    {
        // if(_ldecay != 0)
        //     d4.data[j] = decayL2(d4.data[j], _ldecay);

        d4.data[j] += Optional(o3[j], e4, &d4.momentum[j]); //SGD(o3[j], e4); //Momentum(o3[j], e4, &d4.momentum[j]);
    }

    d4.bias += Optional(1, e4, &d4.bias_momentum); //SGD(1, e4); //Momentum(1, e4, &d4.bias_momentum);

    // done, return forward prop output
    return output;
}

float rmseDiscriminator(const uint start, const uint end)
{
    float squaremean = 0;
    for(uint i = start; i < end; i++)
    {
        const float r = 1 - doDiscriminator(&digest[i][0], NO_LEARN);
        squaremean += r*r;
    }
    squaremean /= DATA_SIZE;
    return sqrt(squaremean);
}

void loadDataset(const char* file)
{
    const time_t st = time(0);

    // read training data [every input is truncated to 256 characters]
    FILE* f = fopen(file, "r");
    if(f)
    {
        char line[MESSAGE_SIZE];
        uint index = 0;
        while(fgets(line, MESSAGE_SIZE, f) != NULL)
        {
            char* pos = strchr(line, '\n');
            if(pos != NULL)
                *pos = '\0';
            uint i = 0;
            char* w = strtok(line, " ");
            
            while(w != NULL)
            {
                digest[index][i] = getWordNorm(w); //normalise
                w = strtok(NULL, " ");
                i++;
            }

            index++;
            if(index == DATA_SIZE)
                break;
        }
        fclose(f);
    }

    printf("Training Data Loaded.\n");
    printf("Time Taken: %.2f mins\n\n", ((double)(time(0)-st)) / 60.0);
}

float trainDataset(const uint start, const uint end)
{
    float rmse = 0;

    // train discriminator
    const time_t st = time(0);
    for(int j = 0; j < TRAINING_LOOPS; j++)
    {
        for(int i = start; i < end; i++)
        {
            // train discriminator on data
            doDiscriminator(&digest[i][0], 1);

            // detrain discriminator on random word sequences 
            float output[DIGEST_SIZE] = {0};
            const int len = uRand(1, DIGEST_SIZE);
            for(int i = 0; i < len; i++)
                output[i] = (((double)uRand(0, TABLE_SIZE))/TABLE_SIZE_H)-1.0; //uRandWeight(-1, 1);
            doDiscriminator(&output[0], 0);

            if(_log == 1)
                printf("Training Iteration (%u / %u) [%u / %u]\n RAND | REAL\n", i+1, DATA_SIZE, j+1, TRAINING_LOOPS);

            if(_log == 1)
            {
                for(int k = 0; k < DIGEST_SIZE; k++)
                    printf("%+.2f : %+.2f\n", output[k], digest[i][k]);

                printf("\n");
            }
        }

        rmse = rmseDiscriminator(DATA_SIZE * DATA_TRAIN_PERCENT, DATA_SIZE);
        if(_log == 1)
            printf("RMSE: %f :: %lus\n", rmse, time(0)-st);
        if(_log == 2)
            printf("RMSE:          %.2f :: %lus\n", rmse, time(0)-st);
    }

    // return rmse
    return rmse;
}


//*************************************
// program functions
//*************************************

void consoleAsk()
{
    // what percentage human is this ?
    while(1)
    {
        char str[MESSAGE_SIZE] = {0};
        float nstr[DIGEST_SIZE] = {0};
        printf(": ");
        fgets(str, MESSAGE_SIZE, stdin);
        str[strlen(str)-1] = 0x00; //remove '\n'

        //normalise words
        uint i = 0;
        char* w = strtok(str, " ");
        while(w != NULL)
        {
            nstr[i] = getWordNorm(w);
            w = strtok(NULL, " ");
            i++;
        }

        const float r = doDiscriminator(nstr, NO_LEARN);
        printf("This is %.2f%% (%.2f) Human.\n", r * 100, r);
    }
}

float isHuman(char* str)
{
    float nstr[DIGEST_SIZE] = {0};

    //normalise words
    uint i = 0;
    char* w = strtok(str, " ");
    while(w != NULL)
    {
        nstr[i] = getWordNorm(w);
        printf("> %s : %f\n", w, nstr[i]);
        w = strtok(NULL, " ");
        i++;
    }

    const float r = doDiscriminator(nstr, NO_LEARN);
    return r*100;
}

float rndScentence(const uint silent)
{
    float nstr[DIGEST_SIZE] = {0};
    const int len = uRand(1, DIGEST_SIZE);
    for(int i = 0; i < len; i++)
        nstr[i] = (((double)uRand(0, TABLE_SIZE))/TABLE_SIZE_H)-1.0;

    for(int i = 0; i < DIGEST_SIZE; i++)
    {
        const uint ind = (((double)nstr[i]+1.0)*(double)TABLE_SIZE_H)+0.5;
        if(nstr[i] != 0 && silent == 0)
            printf("%s (%.2f) ", wtable[ind], nstr[i]);
    }

    if(silent == 0)
        printf("\n");

    const float r = doDiscriminator(nstr, NO_LEARN);
    return r*100;
}

uint rndGen(const char* file, const float max)
{
    FILE* f = fopen(file, "w");
    if(f != NULL)
    {
        uint count = 0;
        time_t st = time(0);
        for(int k = 0; k < OUTPUT_QUOTES; NULL)
        {
            float nstr[DIGEST_SIZE] = {0};
            const int len = uRand(1, DIGEST_SIZE);
            for(int i = 0; i < len; i++)
                nstr[i] = (((double)uRand(0, TABLE_SIZE))/TABLE_SIZE_H)-1.0;

            const float r = doDiscriminator(nstr, NO_LEARN);
            if(1-r < max)
            {
                for(int i = 0; i < DIGEST_SIZE; i++)
                {
                    const uint ind = (((double)nstr[i]+1.0)*(double)TABLE_SIZE_H)+0.5;
                    if(nstr[i] != 0)
                    {
                        fprintf(f, "%s ", wtable[ind]);
                        if(_log == 1)
                            printf("%s ", wtable[ind]);
                    }
                }
                
                k++;
                count++;
                fprintf(f, "\n");
                if(_log == 1)
                    printf("\n");
            }

            if(time(0) - st > 9) // after 9 seconds
            {
                if(count < 450)
                {
                    printf(":: Terminated at a RPS of %u/50 per second.\n", count/9);
                    return 0; // if the output rate was less than 50 per second, just quit.
                }
                
                count = 0;
                st = time(0);
            }
        }

        fclose(f);
    }

    return 1;
}

float hasFailed(const uint resolution)
{
    int failvariance = 0;
    for(int i = 0; i < 100*resolution; i++)
    {
        const float r = rndScentence(1);
        if(r < 50)
            failvariance++;
    }
    if(resolution == 1)
        return failvariance;
    else
        return (double)failvariance / (double)resolution;
}

uint huntBestWeights(float* rmse)
{
    *rmse = 0;
    float fv = 0;
    float min = 70;
    const float max = 96.0;
    float highest = 0;
    time_t st = time(0);
    while(fv < min || fv > max) //we want random string to fail at-least 70% of the time / but we don't want it to fail all of the time
    {
        newSRAND(); //kill any predictability in the random generator

        _loptimiser = uRand(0, 4);
        _lrate      = uRandFloat(0.001, 0.03);
        //_ldecay     = uRandFloat(0.1, 0.0001);
        _ldropout   = uRandFloat(0, 0.3);
        if(_loptimiser == 1 || _loptimiser == 2)
            _lmomentum  = uRandFloat(0.1, 0.9);
        if(_loptimiser == 4)
            _lrmsalpha  = uRandFloat(0.2, 0.99);

        resetPerceptrons();
        *rmse = trainDataset(0, DATA_SIZE * DATA_TRAIN_PERCENT);

        fv = hasFailed(100);
        if(fv <= max && fv > highest)
            highest = fv;

        if(time(0) - st > 540) //If taking longer than 3 mins just settle with the highest logged in that period
        {
            min = highest;
            highest = 0;
            st = time(0);
            printf("Taking too long, new target: %.2f\n", min);
        }

        printf("RMSE: %f / Fail: %.2f\n", *rmse, fv);
    }
    return fv; // fail variance
}

void rndBest()
{
    _log = 2;
    loadDataset("botmsg.txt");

    // load the last lowest fv target
    FILE* f = fopen("gs.dat", "r");
    while(f == NULL)
    {
        f = fopen("gs.dat", "r");
        usleep(1000); //1ms
    }
    float min = 0;
    while(fread(&min, 1, sizeof(float), f) != sizeof(float))
        usleep(1000);
    fclose(f);
    printf("Start fail variance: %.2f\n\n", min);

    // find a new lowest fv target
    while(1)
    {
        float rmse = 0;
        const time_t st = time(0);
        float fv = 0;
        const float max = 96.0;
        while(fv < min || fv > max) //we want random string to fail at-least some percent of the time more than 50% preferably
        {
            newSRAND(); //kill any predictability in the random generator

            _loptimiser = uRand(0, 4);
            _lrate      = uRandFloat(0.001, 0.03);
            //_ldecay     = uRandFloat(0.1, 0.0001);
            _ldropout   = uRandFloat(0, 0.3);
            if(_loptimiser == 1 || _loptimiser == 2)
                _lmomentum  = uRandFloat(0.1, 0.9);
            if(_loptimiser == 4)
                _lrmsalpha  = uRandFloat(0.2, 0.99);
            printf("Optimiser:     %u\n", _loptimiser);
            printf("Learning Rate: %f\n", _lrate);
            //printf("Decay:         %f\n", _ldecay);
            printf("Dropout:       %f\n", _ldropout);
            if(_loptimiser == 1 || _loptimiser == 2)
                printf("Momentum:      %f\n", _lmomentum);
            else if(_loptimiser == 4)
                printf("RMSProp Alpha: %f\n", _lrmsalpha);

            printf("~\n");

            resetPerceptrons();
            rmse = trainDataset(0, DATA_SIZE * DATA_TRAIN_PERCENT);
            
            const time_t st2 = time(0);
            fv = hasFailed(100);
            printf("Fail Variance: %.2f :: %lus\n---------------\n", fv, time(0)-st2);
        }

        // this allows multiple processes to compete on the best weights
        f = fopen("gs.dat", "r+");
        while(f == NULL)
        {
            f = fopen("gs.dat", "r+");
            usleep(1000); //1ms
        }
        while(fread(&min, 1, sizeof(float), f) != sizeof(float))
            usleep(1000);
        if(min < fv)
        {
            while(flock(fileno(f), LOCK_EX) == -1)
                usleep(1000);
            while(fseek(f, 0, SEEK_SET) < 0)
                usleep(1000);
            while(fwrite(&fv, 1, sizeof(float), f) != sizeof(float))
                usleep(1000);
            flock(fileno(f), LOCK_UN);

            min = fv;

            saveWeights();
        }
        fclose(f);

        // keep a log of the best configurations
        FILE* f = fopen("best_configs.txt", "a");
        if(f != NULL)
        {
            while(flock(fileno(f), LOCK_EX) == -1)
                usleep(1000);

            fprintf(f, "Fail Variance: %f\n", fv);
            fprintf(f, "RMSE: %f\n", rmse);
            fprintf(f, "Optimiser: %u\n", _loptimiser);
            fprintf(f, "L-Rate: %f\n", _lrate);
            fprintf(f, "Dropout: %f\n", _ldropout);
            if(_loptimiser == 1 || _loptimiser == 2)
                fprintf(f, "Momentum: %f\n", _lmomentum);
            else if(_loptimiser == 4)
                fprintf(f, "RMS Alpha: %f\n", _lrmsalpha);
            fprintf(f, "\n");

            flock(fileno(f), LOCK_UN);
            fclose(f);
        }

        // done    
        const double time_taken = ((double)(time(0)-st)) / 60.0;
        printf("Time Taken: %.2f mins\n\n", time_taken);

        if(fv >= 99.0 || min >= max)
            exit(0);
    }
    exit(0);
}

void bestSetting(const float min)
{
    _log = 2;
    loadDataset("botmsg.txt");

    float a0=0, a1=0, a2=0, a3=0, a4=0, a5=0, a6=0, a7=0;
    uint count = 0, c1 = 0, c2 = 0;

    uint oc[5] = {0};

    // find a new lowest fv target
    while(1)
    {
        float rmse = 0;
        const time_t st = time(0);
        float fv = 0;
        const float max = 96.0;
        while(fv < min || fv > max) //we want random string to fail at-least some percent of the time more than 50% preferably
        {
            newSRAND(); //kill any predictability in the random generator

            _loptimiser = uRand(0, 4);
            _lrate      = uRandFloat(0.001, 0.03);
            //_ldecay     = uRandFloat(0.1, 0.0001);
            _ldropout   = uRandFloat(0, 0.3);
            if(_loptimiser == 1 || _loptimiser == 2)
                _lmomentum  = uRandFloat(0.1, 0.9);
            if(_loptimiser == 4)
                _lrmsalpha  = uRandFloat(0.2, 0.99);

            resetPerceptrons();
            rmse = trainDataset(0, DATA_SIZE * DATA_TRAIN_PERCENT);

            const time_t st2 = time(0);
            fv = hasFailed(100);
            printf("Fail Variance: %.2f :: %lus\n---------------\n", fv, time(0)-st2);
        }

        a0 += fv;
        a1 += rmse;
        a2 += _loptimiser;
        a3 += _lrate;
        a4 += _ldropout;
        count++;
        oc[_loptimiser]++;

        if(_loptimiser == 1 || _loptimiser == 2)
        {
            a5 += _lmomentum;
            c1++;
        }
        else if(_loptimiser == 4)
        {
            a6 += _lrmsalpha;
            c2++;
        }

        // keep a log of the average
        FILE* f = fopen("best_average.txt", "w");
        if(f != NULL)
        {
            while(flock(fileno(f), LOCK_EX) == -1)
                usleep(1000);

            fprintf(f, "Iterations: %u\n", count);
            fprintf(f, "Fail Variance: %f\n", a0/count);
            fprintf(f, "RMSE: %f\n", a1/count);
            fprintf(f, "L-Rate: %f\n", a3/count);
            fprintf(f, "Dropout: %f\n", a4/count);
            if(a5 != 0)
                fprintf(f, "Momentum: %f / %u\n", a5/c1, c1);
            if(a6 != 0)
                fprintf(f, "RMS Alpha: %f / %u\n", a6/c2, c2);
            fprintf(f, "Optimiser: %f\n", a2/count);
            for(uint i = 0; i < 5; i++)
                fprintf(f, "\nOptimiser-%u: %u\n", i, oc[i]);
            fprintf(f, "\n");

            flock(fileno(f), LOCK_UN);
            fclose(f);
        }

        // keep a log of the best configurations
        f = fopen("best_configs.txt", "a");
        if(f != NULL)
        {
            while(flock(fileno(f), LOCK_EX) == -1)
                usleep(1000);

            fprintf(f, "Fail Variance: %f\n", fv);
            fprintf(f, "RMSE: %f\n", rmse);
            fprintf(f, "Optimiser: %u\n", _loptimiser);
            fprintf(f, "L-Rate: %f\n", _lrate);
            fprintf(f, "Dropout: %f\n", _ldropout);
            if(_loptimiser == 1 || _loptimiser == 2)
                fprintf(f, "Momentum: %f\n", _lmomentum);
            else if(_loptimiser == 4)
                fprintf(f, "RMS Alpha: %f\n", _lrmsalpha);
            fprintf(f, "\n");

            flock(fileno(f), LOCK_UN);
            fclose(f);
        }

        // done    
        const double time_taken = ((double)(time(0)-st)) / 60.0;
        printf("\nbest_average.txt updated ; Time Taken: %.2f mins\n\n", time_taken);
    }
    exit(0);
}

void resetState(const float min)
{
    remove("weights.dat");
            
    FILE* f = fopen("gs.dat", "w");
    while(f == NULL)
    {
        f = fopen("gs.dat", "w");
        usleep(1000);
    }
    while(fwrite(&min, 1, sizeof(float), f) != sizeof(float))
        usleep(1000);
    fclose(f);
}


//*************************************
// program entry point
//*************************************

int main(int argc, char *argv[])
{
    // init discriminator
    createPerceptrons();

    // load lookup table
    loadTable("botdict.txt");

    // are we issuing any commands?
    if(argc == 3)
    {
        if(strcmp(argv[1], "retrain") == 0)
        {
            _log = 1;
            resetState(70);
            loadDataset(argv[2]);
            trainDataset(0, DATA_SIZE);
            const time_t st2 = time(0);
            const float fv = hasFailed(100);
            printf("Fail Variance: %.2f :: %lus\n---------------\n", fv, time(0)-st2);
            saveWeights();
            exit(0);
        }

        if(strcmp(argv[1], "bestset") == 0)
            bestSetting(atoi(argv[2]));

        if(strcmp(argv[1], "gen") == 0)
        {
            _log = 1;
            printf("Brute forcing with an error of: %s\n\n", argv[2]);
            loadWeights();
            rndGen("out.txt", atof(argv[2]));
            exit(0);
        }

        if(strcmp(argv[1], "reset") == 0)
        {
            resetState(atoi(argv[2]));
            printf("Weights and multi-process descriptor reset.\n");
            exit(0);
        }
    }

    if(argc == 2)
    {
        if(strcmp(argv[1], "retrain") == 0)
        {
            _log = 1;
            resetState(70);
            loadDataset("botmsg.txt");
            trainDataset(0, DATA_SIZE);
            const time_t st2 = time(0);
            const float fv = hasFailed(100);
            printf("Fail Variance: %.2f :: %lus\n---------------\n", fv, time(0)-st2);
            saveWeights();
            exit(0);
        }

        if(strcmp(argv[1], "bestset") == 0)
            bestSetting(70);

        if(strcmp(argv[1], "best") == 0)
            rndBest();
        
        if(strcmp(argv[1], "reset") == 0)
        {
            resetState(70);
            printf("Weights and multi-process descriptor reset.\n");
            exit(0);
        }

        if(strcmp(argv[1], "check") == 0)
        {
            FILE* f = fopen("gs.dat", "r");
            while(f == NULL)
            {
                f = fopen("gs.dat", "r");
                usleep(1000); //1ms
            }
            float fv = 0;
            while(fread(&fv, 1, sizeof(float), f) != sizeof(float))
                usleep(1000);
            fclose(f);

            printf("Current weights have a fail variance of %f.\n", fv);

            struct stat st;
            const int sr = stat("weights.dat", &st);
            setlocale(LC_NUMERIC, "");
            if(sr == 0 && st.st_size > 0)
                printf("%'.0f kb / %'.2f mb / %'.2f gb\n", (double)st.st_size / 1000, ((((double)st.st_size) / 1000) / 1000), ((((double)st.st_size) / 1000) / 1000) / 1000);
            else
                printf("weights.dat is 0 bytes.\n");
            exit(0);
        }

        ///////////////////////////
        loadWeights();
        ///////////////////////////

        if(strcmp(argv[1], "ask") == 0)
            consoleAsk();

        if(strcmp(argv[1], "rnd") == 0)
        {
            newSRAND();

            printf("> %.2f\n", rndScentence(0));
            exit(0);
        }

        if(strcmp(argv[1], "gen") == 0)
        {
            _log = 1;
            rndGen("out.txt", 0.5);
            exit(0);
        }

        if(strcmp(argv[1], "rndloop") == 0)
        {
            newSRAND();
            while(1)
                printf("> %.2f\n\n", rndScentence(0));
        }

        char in[MESSAGE_SIZE] = {0};
        snprintf(in, MESSAGE_SIZE, "%s", argv[1]);
        printf("%.2f\n", isHuman(in));
        exit(0);
    }

    // no commands ? run as service
    loadWeights();

    // main loop
    printf("Running ! ...\n\n");
    while(1)
    {
        if(countLines("botmsg.txt") >= DATA_SIZE)
        {
            timestamp();
            const time_t st = time(0);
            memset(&wtable, 0x00, TABLE_SIZE_MAX*WORD_SIZE);
            resetState(70);
            loadTable("botdict.txt");
            loadDataset("botmsg.txt");
            clearFile("botmsg.txt");

            float rmse = 0;
            uint fv = huntBestWeights(&rmse);
            while(rndGen("out.txt", 0.2) == 0)
                fv = huntBestWeights(&rmse);

            saveWeights();
            printf("Just generated a new dataset.\n");
            timestamp();
            const double time_taken = ((double)(time(0)-st)) / 60.0;
            printf("Time Taken: %.2f mins\n\n", time_taken);

            FILE* f = fopen("portstat.txt", "w");
            if(f != NULL)
            {
                const time_t ltime = time(0);
                setlocale(LC_NUMERIC, "");
                fprintf(f, "Trained with an RMSE of %f and Fail Variance of %u (higher is better) on;\n%sTime Taken: %.2f minutes\nDigest size: %'u\n", rmse, fv, asctime(localtime(&ltime)), time_taken, DATA_SIZE);
                fprintf(f, "L-Rate: %f\n", _lrate);
                //fprintf(f, "Decay: %f\n", _ldecay);
                fprintf(f, "Dropout: %f\n", _ldropout);
                if(_loptimiser == 1 || _loptimiser == 2)
                    fprintf(f, "Momentum: %f\n", _lmomentum);
                else if(_loptimiser == 4)
                    fprintf(f, "RMS Alpha: %f\n", _lrmsalpha);
                fprintf(f, "Optimiser: %u\n\n", _loptimiser);
                fprintf(f, "I is very smort and I hab big brain of %'u perceptronic neurons with %'u configurable weights.\n", FIRSTLAYER_SIZE + HIDDEN_SIZE + HIDDEN_SIZE + 1, FIRSTLAYER_SIZE*(DIGEST_SIZE+1) + HIDDEN_SIZE*(FIRSTLAYER_SIZE+1) + HIDDEN_SIZE*(HIDDEN_SIZE+1) + (HIDDEN_SIZE+1));
                fclose(f);
            }
        }

        sleep(9);
    }

    // done
    return 0;
}

