from os.path import join
import shutil
import os


def log_poses_origin(r_dir, r_names, render_ts, render_qvecs, renderer):
    if renderer == 'nerf':
        width = 5
        r_names = [str(idx).zfill(width) for idx in range(len(r_names))]

    with open(join(r_dir, 'rendered_views.txt'), 'w') as rv:
        for i in range(len(r_names)):
            line_data = [r_names[i], *tuple(render_ts[i]), *tuple(render_qvecs[i])]
            line = " ".join(map(str, line_data))
            rv.write(line+'\n')

def log_poses(r_dir, r_names, render_ts, render_qvecs, K, w, h, renderer):
    fx = K[0, 0]# / 8.0
    fy = K[1, 1]# / 8.0
    cx = K[0, 2]# / 8.0
    cy = K[1, 2]# / 8.0
    w = w # / 8.0
    h = h # / 8.0
    if renderer == 'nerf':
        width = 5
        r_names = [str(idx).zfill(width) for idx in range(len(r_names))]

    with open(join(r_dir, 'rendered_views.txt'), 'w') as rv:
        for i in range(len(r_names)):
            line_data = [r_names[i], *tuple(render_qvecs[i]), *tuple(render_ts[i])]
            line = " ".join(map(str, line_data))
            rv.write(line+'\n')

    with open(join(r_dir, 'rendered_intrinsics.txt'), 'w') as rv:
        for r_name in r_names:
            line = f"{r_name} PINHOLE {w} {h} {fx} {fy} {cx} {cy}"
            rv.write(line + '\n')


def split_to_beam_folder(r_dir, n_beams, r_names_per_beam_q_idx, create_beams=False):
    for beam_i in range(n_beams):
        beam_dir = join(r_dir, f'beam_{beam_i}')
        if create_beams:
            os.makedirs(beam_dir, exist_ok=True)
        beam_names = r_names_per_beam_q_idx[beam_i]
        for b_name in beam_names:
            src = join(r_dir, b_name+'.png')
            dst = join(beam_dir, b_name+'.png')
            shutil.move(src, dst)
