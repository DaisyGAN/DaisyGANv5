/*
--------------------------------------------------
    91f7d09794d8da29f028e77df49d4907
    https://github.com/DaisyGAN/
--------------------------------------------------
    DaisyGANv5

    Technically not a generative adversarial network anymore.
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

#define uint uint32_t
#define NO_LEARN -2

///

#define TABLE_SIZE_MAX 80000
#define DIGEST_SIZE 16
#define WORD_SIZE 256 //32
#define MESSAGE_SIZE WORD_SIZE*DIGEST_SIZE

///

#define FAST_PREDICTABLE_MODE
#define DATA_SIZE 2411
#define OUTPUT_QUOTES 33333
#define FIRSTLAYER_SIZE 128
#define HIDDEN_SIZE 128
#define TRAINING_LOOPS 1
float       _lrate      = 0.03;
float       _ldecay     = 0.0005;
float       _ldropout   = 0.2;
uint        _lbatches   = 1;
uint        _loptimiser = 4;
float       _lmomentum  = 0.1;
float       _lrmsalpha  = 0.2; //0.99
const float _lgain      = 1.0;

// #define FAST_PREDICTABLE_MODE
// #define DATA_SIZE 2411
// #define OUTPUT_QUOTES 33333
// #define FIRSTLAYER_SIZE 256
// #define HIDDEN_SIZE 256
// #define TRAINING_LOOPS 1
// float       _lrate      = 0.03;
// float       _ldecay     = 0.0005;
// float       _ldropout   = 0.2;
// uint        _lbatches   = 3;
// uint        _loptimiser = 4;
// float       _lmomentum  = 0.1;
// float       _lrmsalpha  = 0.2; //0.99
// const float _lgain      = 1.0;

// this is not the vegetarian option
// #define FAST_PREDICTABLE_MODE
// #define DATA_SIZE 2411
// #define OUTPUT_QUOTES 33333
// #define FIRSTLAYER_SIZE 512
// #define HIDDEN_SIZE 1024
// #define TRAINING_LOOPS 1
// float       _lrate      = 0.01;
// float       _ldecay     = 0.0005;
// float       _ldropout   = 0.3;
// uint        _lbatches   = 180;
// uint        _loptimiser = 1;
// float       _lmomentum  = 0.1;
// float       _lrmsalpha  = 0.2;
// const float _lgain      = 1.0;

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
    return ( (rv / RAND_MAX) * (max-min) ) + min;
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
    return ( (rv / RAND_MAX) * (max-min) ) + min;
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
        const float rv2 = ( (rv / RAND_MAX) * (max-min) ) + min;
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
        const float rv2 = ( (rv / RAND_MAX) * (max-min) ) + min;
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
    return ( ((float)rv / RAND_MAX) * (max-min) ) + min; //(rand()%(max-min))+min;
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
    return ( ((float)rv / RAND_MAX) * (max-min) ) + min; //(rand()%(max-min))+min;
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

void createPerceptron(ptron* p, const uint weights)
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

    for(uint i = 0; i < weights; i++)
    {
        p->data[i] = qRandWeight(-1, 1);
        p->momentum[i] = 0;
    }

    p->bias = qRandWeight(-1, 1);
    p->bias_momentum = 0;
}

void resetPerceptron(ptron* p)
{
    for(uint i = 0; i < p->weights; i++)
    {
        p->data[i] = qRandWeight(-1, 1);
        p->momentum[i] = 0;
    }

    p->bias = qRandWeight(-1, 1);
    p->bias_momentum = 0;
}

void createPerceptrons()
{
    for(int i = 0; i < FIRSTLAYER_SIZE; i++)
        createPerceptron(&d1[i], DIGEST_SIZE);
    for(int i = 0; i < HIDDEN_SIZE; i++)
        createPerceptron(&d2[i], FIRSTLAYER_SIZE);
    for(int i = 0; i < HIDDEN_SIZE; i++)
        createPerceptron(&d3[i], HIDDEN_SIZE);
    createPerceptron(&d4, HIDDEN_SIZE);
}

void resetPerceptrons()
{
    for(int i = 0; i < FIRSTLAYER_SIZE; i++)
        resetPerceptron(&d1[i]);
    for(int i = 0; i < HIDDEN_SIZE; i++)
        resetPerceptron(&d2[i]);
    for(int i = 0; i < HIDDEN_SIZE; i++)
        resetPerceptron(&d3[i]);
    resetPerceptron(&d4);
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
    // const float ret = _lrate * ( input * (input + _lmomentum * momentum[0]) ) + (_lmomentum * momentum[0]);
    // momentum[0] = input;
    // return ret;

    const float ret = _lrate * ( error * (input + _lmomentum * momentum[0]) ) + (_lmomentum * momentum[0]);
    momentum[0] = input;
    return ret;
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
        return ADAGrad(input, error, momentum);
    else if(_loptimiser == 3)
        return RMSProp(input, error, momentum);
    else if(_loptimiser == 4)
        return Nesterov(input, error, momentum);
    
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

    float e4 = _lgain * o4 * (1-o4) * error;

    // layer 3 (output)
    float ler = 0;
    for(int j = 0; j < d4.weights; j++)
        ler += d4.data[j] * e4;
    ler += d4.bias * e4;
    
    for(int i = 0; i < HIDDEN_SIZE; i++)
        e3[i] = _lgain * o3[i] * (1-o3[i]) * ler;

    // layer 2
    ler = 0;
    for(int i = 0; i < HIDDEN_SIZE; i++)
    {
        for(int j = 0; j < d3[i].weights; j++)
            ler += d3[i].data[j] * e3[i];
        ler += d3[i].bias * e3[i];
        
        e2[i] = _lgain * o2[i] * (1-o2[i]) * ler;
    }

    // layer 1
    ler = 0;
    float k = 0;
    int ki = 0;
    for(int i = 0; i < FIRSTLAYER_SIZE; i++)
    {
        for(int j = 0; j < d2[i].weights; j++)
            ler += d2[i].data[j] * e2[i];
        ler += d2[i].bias * e2[i];
        
        int k0 = 0;
        if(k != 0)
            k0 = 1;
        k += _lgain * o1[i] * (1-o1[i]) * ler;
        if(k0 == 1)
        {
            e1[ki] = k / 2;
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

float rmseDiscriminator()
{
    float squaremean = 0;
    for(int i = 0; i < DATA_SIZE; i++)
    {
        const float r = 1 - doDiscriminator(&digest[i][0], NO_LEARN);
        squaremean += r*r;
    }
    squaremean /= DATA_SIZE;
    return sqrt(squaremean);
}

void loadDataset(const char* file)
{
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
}

float trainDataset()
{
    float rmse = 0;

    // train discriminator
    const time_t st = time(0);
    for(int j = 0; j < TRAINING_LOOPS; j++)
    {
        for(int i = 0; i < DATA_SIZE; i++)
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

        rmse = rmseDiscriminator();
        if(_log == 1 || _log == 2)
            printf("RMSE: %f :: %lus\n", rmse, time(0)-st);
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
                if(count < 900)
                    return 0; // if the output rate was less than 100 per second, just quit.
                
                count = 0;
                st = time(0);
            }
        }

        fclose(f);
    }

    return 1;
}

float findBest(const uint maxopt)
{
    float lowest_low = 999999999;
    for(uint i = 0; i <= maxopt; i++)
    {
        _loptimiser = i;
        if(_log == 2)
            printf("\nOptimiser: %u\n", _loptimiser);

        const time_t st = time(0);
        float mean = 0, low = 9999999, high = 0;
        for(uint j = 0; j < 3; j++)
        {
            resetPerceptrons();
            const float rmse = trainDataset();
            mean += rmse;
            if(rmse < low)
                low = rmse;
            if(rmse > high)
                high = rmse;
            if(rmse > 0 && rmse < lowest_low)
            {
                lowest_low = rmse;
                saveWeights();
            }
        }
        if(_log == 2)
        {
            printf("Lo  RMSE:   %f\n", low);
            printf("Hi  RMSE:   %f\n", high);
            printf("Avg RMSE:   ~ %f\n", mean / 6);
            printf("RMSE Delta: %f\n", high-low);
            printf("Time Taken: %.2f mins\n", ((double)(time(0)-st)) / 60.0);
        }
    }
    if(_log == 2)
        printf("\nThe dataset with an RMSE of %f was saved to weights.dat\n\n", lowest_low);
    return lowest_low;
}

uint hasFailed()
{
    int failvariance = 0;
    for(int i = 0; i < 100; i++)
    {
        const float r = rndScentence(1);
        if(r < 50)
            failvariance++;
    }
    return failvariance;
}

uint huntBestWeights(float* rmse)
{
    *rmse = 0;
    uint fv = 0;
    uint min = 70;
    const uint max = 95;
    uint highest = 0;
    time_t st = time(0);
    while(fv < min || fv > max) //we want random string to fail at-least 70% of the time / but we don't want it to fail all of the time
    {
        newSRAND(); //kill any predictability in the random generator

        _lrate      = uRandFloat(0.001, 0.03);
        _ldropout   = uRandFloat(0.2, 0.3);
        _lmomentum  = uRandFloat(0.1, 0.9);
        _lrmsalpha  = uRandFloat(0.2, 0.99);

        *rmse = findBest(1);

        loadWeights();
        fv = hasFailed();
        if(fv <= max && fv > highest)
            highest = fv;

        if(time(0) - st > 540) //If taking longer than 3 mins just settle with the highest logged in that period
        {
            min = highest;
            highest = 0;
            st = time(0);
            printf("Taking too long, new target: %u\n", min);
        }

        printf("RMSE: %f / Fail: %u\n", *rmse, fv);
    }
    return fv; // fail variance
}

void rndBest(const uint min)
{
    _log = 2;
    remove("weights.dat");
    loadDataset("botmsg.txt");

    const time_t st = time(0);
    uint fv = 0;
    while(fv < min || fv > 95) //we want random string to fail at-least 70% of the time
    {
        newSRAND(); //kill any predictability in the random generator

        _lrate      = uRandFloat(0.001, 0.03);
        _ldecay     = uRandFloat(0.1, 0.0001);
        _ldropout   = uRandFloat(0.2, 0.3);
        _lmomentum  = uRandFloat(0.1, 0.9);
        _lrmsalpha  = uRandFloat(0.2, 0.99);
        printf("Learning Rate: %f\n", _lrate);
        printf("Decay:         %f\n", _ldecay);
        printf("Dropout:       %f\n", _ldropout);
        printf("Momentum:      %f\n", _lmomentum);
        printf("RMSProp Alpha: %f\n", _lrmsalpha);

        findBest(4);

        loadWeights();
        fv = hasFailed();
        printf("Fail Variance: %u\n\n", fv);
    }

    const double time_taken = ((double)(time(0)-st)) / 60.0;
    printf("Time Taken: %.2f mins\n\n", time_taken);
    exit(0);
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
            remove("weights.dat");
            loadDataset(argv[2]);
            trainDataset();
            saveWeights();
            exit(0);
        }

        if(strcmp(argv[1], "gen") == 0)
        {
            _log = 1;
            printf("Brute forcing with an error of: %s\n\n", argv[2]);
            loadWeights();
            rndGen("out.txt", atof(argv[2]));
            exit(0);
        }

        if(strcmp(argv[1], "rndbest") == 0)
            rndBest(atoi(argv[2]));
    }

    if(argc == 2)
    {
        if(strcmp(argv[1], "retrain") == 0)
        {
            _log = 1;
            remove("weights.dat");
            loadDataset("botmsg.txt");
            trainDataset();
            saveWeights();
            exit(0);
        }

        if(strcmp(argv[1], "rndbest") == 0)
            rndBest(70);

        if(strcmp(argv[1], "best") == 0)
        {
            newSRAND(); //kill any predictability in the random generator

            _log = 2;
            remove("weights.dat");
            loadDataset("botmsg.txt");
            findBest(4);

            loadWeights();
            const uint fv = hasFailed();
            printf("Fail Variance: %u\n\n", fv);
            
            exit(0);
        }

        loadWeights();

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
            loadTable("botdict.txt");
            loadDataset("botmsg.txt");
            clearFile("botmsg.txt");

            _lrate      = uRandFloat(0.001, 0.03);
            _ldecay     = uRandFloat(0.1, 0.0001);
            _ldropout   = uRandFloat(0.2, 0.3);
            _lmomentum  = uRandFloat(0.1, 0.9);
            _lrmsalpha  = uRandFloat(0.2, 0.99);

            float rmse = 0;
            uint fv = huntBestWeights(&rmse);
            while(rndGen("out.txt", 0.1) == 0)
                fv = huntBestWeights(&rmse);

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
                fprintf(f, "Decay: %f\n", _lrate);
                fprintf(f, "Dropout: %f\n", _ldropout);
                fprintf(f, "Momentum: %f\n", _lmomentum);
                fprintf(f, "Alpha: %f\n\n", _lrmsalpha);
                fprintf(f, "I is very smort and I hab big brain of %'u perceptronic neurons with %'u configurable weights.\n", FIRSTLAYER_SIZE + HIDDEN_SIZE + HIDDEN_SIZE + 1, FIRSTLAYER_SIZE*(DIGEST_SIZE+1) + HIDDEN_SIZE*(FIRSTLAYER_SIZE+1) + HIDDEN_SIZE*(HIDDEN_SIZE+1) + (HIDDEN_SIZE+1));
                fclose(f);
            }
        }

        sleep(9);
    }

    // done
    return 0;
}

