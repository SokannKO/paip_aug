import os
from . import html
from . import sec2dhms
import GPUtil

def vis_html(cfg, res_dict):
    html_path = cfg.CKPT_DIR
    webpage = html.HTML(html_path, cfg.CKPT_DIR, refresh=1)

    webpage.add_header('Status graph')
    es1, es2, es3 = [], [], []

    es1.append('LA_plot.png')
    es2.append('')
    es3.append('LA_plot.png')

    d, h, m, s = sec2dhms(cfg.TRAIN.timestamp[4])
    time_left = '%ddays %dhours %dminutes %dseconds' % (d, h, m, s)
    datasets = ['TRAIN : '+str(cfg.TRAIN.num_data), 'VAL : '+str(cfg.VAL.num_data)]
    es1.append(['Estimated time remaining', time_left])
    es2.append(['Training Type', cfg.MODEL.type])
    es3.append(['Dataset', datasets])

    GPUs = GPUtil.getGPUs()
    gpu_list = ['{0}: {1}\t{2}°C\t{3}MB / {4}MB'.format(
        f.id,f.name,f.temperature,f.memoryUsed,f.memoryTotal) for f in GPUs]

    es1.append('GPU usage')
    es2.append(gpu_list)
    es3.append('')

    webpage.add_status_table(es1, es2, es3)

    res_arr = res_dict
    ep = len(res_arr['train'])
    for e in range(1,ep+1,1):
        webpage.add_text('[TRAIN]['+str(e)+'] '+str(res_arr['train'][e])
                         +'\t'+
                         '[VAL]['+str(e)+'] '+str(res_arr['val'][e]))

    '''
    for e in range(ep, 0, -1):
        webpage.add_text('[TRAIN]')
    '''
    webpage.save()



if __name__ == '__main__':  # we show an example usage here.
    webpage = html.HTML('tmp/', 'test_html', refresh=1)

    epoch = 5
    ims, txts, links = [], [], []
    for n in range(1):
        ims.append('image_%d.png' % n)
        txts.append('text_%d' % n)
        links.append('image_%d.png' % n)
    webpage.add_images(ims, txts, links)

    for e in range(epoch, 0, -1):
        webpage.add_text('hello world')
    webpage.save()


