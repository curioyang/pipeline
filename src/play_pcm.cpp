#include "play_pcm.h"
#include "alsa/asoundlib.h"
#include "alsa/pcm.h"

static snd_pcm_t *handle;
static snd_pcm_hw_params_t *params;
static snd_pcm_uframes_t frames;

int initPlayer(int sample_rate,int channels,int frame_samples,int bits_per_sample){
    int rc;
    unsigned int val;
    int dir = 0;
	int ret = -1;

    rc = snd_pcm_open(&handle, "default", SND_PCM_STREAM_PLAYBACK, 0);
    if (rc < 0) {
        printf("unable to open PCM device: %s\n",snd_strerror(rc));
        return -1;
    }

    /* alloc hardware params object */
    snd_pcm_hw_params_alloca(&params);

    /* fill it with default values */
    snd_pcm_hw_params_any(handle, params);

    /* interleaved mode */
    snd_pcm_hw_params_set_access(handle, params, SND_PCM_ACCESS_RW_INTERLEAVED);

	/* signed 16 bit little ending format */
    snd_pcm_hw_params_set_format(handle, params, SND_PCM_FORMAT_S16_LE);

    /* two channels (stereo) */
    snd_pcm_hw_params_set_channels(handle, params, channels);

    /* 44100 bits/second sampling rate (CD quality) */
    snd_pcm_hw_params_set_rate_near(handle, params,(unsigned int*)&sample_rate, &dir);
	//printf("pcm rate: val:%d dir:%d.\n",val,dir);

    /* set period size t 40ms frames */
    frames=frame_samples;
    rc = snd_pcm_hw_params_set_period_size_near(handle, params, &frames, &dir);
	printf("%d rc = %d,pcm frames: frames:%ld dir:%d.\n",__LINE__,rc,frames,dir);

    /* write params to the driver */
    rc = snd_pcm_hw_params(handle, params);
    if (rc < 0) {
        printf("unable to set hw params: %s\n",snd_strerror(rc));
        return -1;
    }
    /* use buffer large enough to hold one period */
    rc = snd_pcm_hw_params_get_period_size(params, &frames, &dir);
	printf("%d,rc = %d,frames:%ld dir:%d.\n",__LINE__,rc,frames,dir);
    return 0;
}

int playPcm(char *decoder_output_buffer)
{
	int ret = 0;
    while(1){
        ret = snd_pcm_writei(handle, decoder_output_buffer, frames);
        if (ret == -EPIPE) {
            /* -EPIPE means underrun */
            // fprintf(stderr, "underrun occured\n");
            snd_pcm_prepare(handle);
        } else if (ret < 0) {
            fprintf(stderr, "error from writei: %s\n", snd_strerror(ret));
        }
        if(ret==0||ret==-EAGAIN||ret==-EPIPE){
            usleep(1000);
            continue;
        }
        break;
    }
    return ret;
}

int deinitPlayer(){
	snd_pcm_drain(handle);
    snd_pcm_close(handle);
	return 0;
}