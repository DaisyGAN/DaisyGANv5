/*
--------------------------------------------------
    91f7d09794d8da29f028e77df49d4907
    https://github.com/DaisyGAN/
--------------------------------------------------
    DaisyGANv5

    Testing on a dataset of 995 quotes.
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

#define uint uint32_t
#define NO_LEARN -2

///

#define TABLE_SIZE_MAX 80000
#define MESSAGE_SIZE 256
#define DIGEST_SIZE 16
#define WORD_SIZE 32

///

#define FAST_PREDICTABLE_MODE
#define DATA_SIZE 995
#define OUTPUT_QUOTES 333
#define FIRSTLAYER_SIZE 256
#define HIDDEN_SIZE 256
#define TRAINING_LOOPS 1
const float _lrate     = 0.03;
const float _dropout   = 0.2;
const uint  _optimiser = 1;
const float _lmomentum = 0.1;
const float _lgain     = 1.0;

// this is not the vegetarian option
// //#define FAST_PREDICTABLE_MODE
// #define DATA_SIZE 995
// #define OUTPUT_QUOTES 333
// #define FIRSTLAYER_SIZE 512
// #define HIDDEN_SIZE 1024
// #define TRAINING_LOOPS 1
// const float _lrate     = 0.03;
// const float _dropout   = 0.5;
// const uint  _optimiser = 1;
// const float _lmomentum = 0.1;
// const float _lgain     = 1.0;

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

void createPerceptron(ptron* p, const uint weights, const uint random)
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

    if(random == 1)
    {
        for(uint i = 0; i < weights; i++)
        {
            p->data[i] = qRandWeight(-1, 1);
            p->momentum[i] = 0;
        }
    }
    else if(random == 0)
    {
        memset(p->data, 0, weights * sizeof(float));
        memset(p->momentum, 0, weights * sizeof(float));
    }

    p->bias = qRandWeight(-1, 1);
    p->bias_momentum = 0;
}


//*************************************
// activation functions
// https://en.wikipedia.org/wiki/Activation_function
// https://www.analyticsvidhya.com/blog/2020/01/fundamentals-deep-learning-activation-functions-when-to-use-them/
// https://adl1995.github.io/an-overview-of-activation-functions-used-in-neural-networks.html
//*************************************

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
    return 1.7159 * atan(0.666666667 * x);
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

float Optional(const float input, const float error, float* momentum)
{
    if(_optimiser == 1)
        return Momentum(input, error, momentum);
    
    return SGD(input, error);
}


//*************************************
// network training functions
//*************************************

float doDiscriminator(const float* input, const float eo)
{
/**************************************
    Forward Prop
**************************************/

    // layer one, inputs (fc)
    float o1[FIRSTLAYER_SIZE];
    for(int i = 0; i < FIRSTLAYER_SIZE; i++)
        o1[i] = lecun_tanh(doPerceptron(input, &d1[i]));

    // layer two, hidden (fc expansion)
    float o2[HIDDEN_SIZE];
    for(int i = 0; i < HIDDEN_SIZE; i++)
        o2[i] = lecun_tanh(doPerceptron(&o1[0], &d2[i]));

    // layer three, hidden (fc)
    float o3[HIDDEN_SIZE];
    
    for(int i = 0; i < HIDDEN_SIZE; i++)
        o3[i] = lecun_tanh(doPerceptron(&o2[0], &d3[i]));

    // layer four, output (fc compression)
    const float output = sigmoid(lecun_tanh(doPerceptron(&o3[0], &d4)));

    if(eo == NO_LEARN)
        return output;

/**************************************
    Backward Prop Error
**************************************/

    float e1[FIRSTLAYER_SIZE];
    float e2[HIDDEN_SIZE];
    float e3[HIDDEN_SIZE];

    // layer 4
    const float error = eo - output;
    float e4 = _lgain * output * (1-output) * error;

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
        if(_dropout != 0 && uRandWeight(0.01, 1) <= _dropout)
            continue;

        for(int j = 0; j < d1[i].weights; j++)
            d1[i].data[j] += Optional(input[j], e1[i], &d1[i].momentum[j]); //SGD(input[j], e1[i]); //Momentum(input[j], e1[i], &d1[i].momentum[j]);

        d1[i].bias += Optional(1, e1[i], &d1[i].bias_momentum); //SGD(1, e1[i]); //Momentum(1, e1[i], &d1[i].bias_momentum);
    }

    // layer 2
    for(int i = 0; i < HIDDEN_SIZE; i++)
    {
        if(_dropout != 0 && uRandWeight(0.01, 1) <= _dropout)
            continue;

        for(int j = 0; j < d2[i].weights; j++)
            d2[i].data[j] += Optional(o1[j], e2[i], &d2[i].momentum[j]); //SGD(o1[j], e2[i]); //Momentum(o1[j], e2[i], &d2[i].momentum[j]);

        d2[i].bias += Optional(1, e2[i], &d2[i].bias_momentum); //SGD(1, e2[i]); //Momentum(1, e2[i], &d2[i].bias_momentum);
    }

    // layer 3
    for(int i = 0; i < HIDDEN_SIZE; i++)
    {
        if(_dropout != 0 && uRandWeight(0.01, 1) <= _dropout)
            continue;
            
        for(int j = 0; j < d3[i].weights; j++)
            d3[i].data[j] += Optional(o2[j], e3[i], &d3[i].momentum[j]); //SGD(o2[j], e3[i]); //Momentum(o2[j], e3[i], &d3[i].momentum[j]);

        d3[i].bias += Optional(1, e3[i], &d3[i].bias_momentum); //SGD(1, e3[i]); //Momentum(1, e3[i], &d3[i].bias_momentum);
    }

    // layer 4
    for(int j = 0; j < d4.weights; j++)
        d4.data[j] += Optional(o3[j], e4, &d4.momentum[j]); //SGD(o3[j], e4); //Momentum(o3[j], e4, &d4.momentum[j]);

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

void trainDataset(const char* file)
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

    printf("Training Data Loaded\n");

    // train discriminator
    for(int j = 0; j < TRAINING_LOOPS; j++)
    {
        for(int i = 0; i < DATA_SIZE; i++)
        {
            // train discriminator on data
            doDiscriminator(&digest[i][0], 1);

            // detrain discriminator on random word sequences 
            float output[DIGEST_SIZE] = {0};
            const int len = uRand(1, DIGEST_SIZE-1);
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

        printf("RMSE: %f\n", rmseDiscriminator());
    }

    // save weights
    saveWeights();
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

float rndScentence()
{
    float nstr[DIGEST_SIZE] = {0};
    const int len = uRand(1, DIGEST_SIZE-1);
    for(int i = 0; i < len; i++)
        nstr[i] = (((double)uRand(0, TABLE_SIZE))/TABLE_SIZE_H)-1.0;

    for(int i = 0; i < DIGEST_SIZE; i++)
    {
        const uint ind = (((double)nstr[i]+1.0)*(double)TABLE_SIZE_H)+0.5;
        if(nstr[i] != 0)
            printf("%s (%.2f) ", wtable[ind], nstr[i]);
    }

    printf("\n");

    const float r = doDiscriminator(nstr, NO_LEARN);
    return r*100;
}

void rndGen(const char* file, const float max)
{
    FILE* f = fopen(file, "w");
    if(f != NULL)
    {
        for(int k = 0; k < OUTPUT_QUOTES; NULL)
        {
            float nstr[DIGEST_SIZE] = {0};
            const int len = uRand(1, DIGEST_SIZE-1);
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
                        printf("%s ", wtable[ind]);
                    }
                }
                
                k++;
                fprintf(f, "\n");
                printf("\n");
            }
        }

        fclose(f);
    }
}


//*************************************
// program entry point
//*************************************

int main(int argc, char *argv[])
{
    // init discriminator
    for(int i = 0; i < FIRSTLAYER_SIZE; i++)
        createPerceptron(&d1[i], DIGEST_SIZE, 1);
    for(int i = 0; i < HIDDEN_SIZE; i++)
        createPerceptron(&d2[i], FIRSTLAYER_SIZE, 1);
    for(int i = 0; i < HIDDEN_SIZE; i++)
        createPerceptron(&d3[i], HIDDEN_SIZE, 1);
    createPerceptron(&d4, HIDDEN_SIZE, 1);

    // load lookup table
    loadTable("tgdict.txt");

    // are we issuing any commands?
    if(argc == 3)
    {
        if(strcmp(argv[1], "retrain") == 0)
        {
            _log = 1;
            remove("weights.dat");
            trainDataset(argv[2]);
            exit(0);
        }

        if(strcmp(argv[1], "gen") == 0)
        {
            printf("Brute forcing with an error of: %s\n\n", argv[2]);
            loadWeights();
            rndGen("out.txt", atof(argv[2]));
            exit(0);
        }
    }

    if(argc == 2)
    {
        if(strcmp(argv[1], "retrain") == 0)
        {
            _log = 1;
            remove("weights.dat");
            trainDataset("tgmsg.txt");
            exit(0);
        }

        loadWeights();

        if(strcmp(argv[1], "ask") == 0)
            consoleAsk();

        if(strcmp(argv[1], "rnd") == 0)
        {
            printf("> %.2f\n", rndScentence());
            exit(0);
        }

        if(strcmp(argv[1], "gen") == 0)
        {
            rndGen("out.txt", 0.5);
            exit(0);
        }

        if(strcmp(argv[1], "rndloop") == 0)
        {
            while(1)
                printf("> %.2f\n\n", rndScentence());
        }

        char in[MESSAGE_SIZE] = {0};
        snprintf(in, MESSAGE_SIZE, "%s", argv[1]);
        printf("%.2f\n", isHuman(in));
        exit(0);
    }

    // no commands ? then I would suppose we are running the generator
    loadWeights();

    // main loop
    printf("Running ! ...\n\n");
    while(1)
    {
        if(countLines("tgmsg.txt") >= DATA_SIZE)
        {
            timestamp();
            const time_t st = time(0);
            loadTable("tgdict.txt");
            trainDataset("tgmsg.txt");
            clearFile("tgmsg.txt");
            rndGen("out.txt", 0.1);
            printf("Just generated a new dataset.\n");
            timestamp();
            printf("Time Taken: %.2f mins\n\n", ((double)(time(0)-st)) / 60.0);
        }

        sleep(9);
    }

    // done
    return 0;
}

