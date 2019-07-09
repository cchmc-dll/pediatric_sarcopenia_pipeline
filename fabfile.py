import glob
import os
import shutil
import tarfile

from fabric import task
from pathlib import PurePosixPath, Path

user = 'dockeruser'
docker_image_name = 'jac241/muscle_segmentation'

docker_build_cmd = f'docker build -t {docker_image_name} .'

home_dir = PurePosixPath('/home') / user
dataset_dir = home_dir / 'data' / 'datasets'
output_dir = home_dir / 'output'
predictions_dir = home_dir / 'predictions'

docker_run_cmd = (
        f'docker run -u $(id -u):$(id -g) --runtime=nvidia '
        f'-v {dataset_dir}:/opt/app/datasets '
        f'-v {output_dir}:/opt/app/output '
        f'-v {predictions_dir}:/opt/app/predictions '
        f'-t {docker_image_name} '
        f'python run_training.py '
)

app_target_dir = home_dir / 'test'

archive_path = Path(os.getcwd(), 'tmp', 'repo.tar.gz')


@task
def upload(c):
    make_and_enter_app_target_dir(c)
    zip_app_locally()
    copy_app_to_remote(c)
    unzip_app_on_remote(c)
    build_docker_image(c)
    run_docker_image(c)

    c.run(f'ls -la {app_target_dir}')


def make_and_enter_app_target_dir(c):
    c.run(f'mkdir -p {app_target_dir}')
    c.run(f'cd {app_target_dir}')


def zip_app_locally():
    with tarfile.open(archive_path, 'w:gz') as archive:
        archive.add('./', exclude=exclude_select_dirs)


def exclude_select_dirs(path):
    try:
        return is_hidden_dir(path) or is_tmp_dir(path)
    except IndexError:
        return False


def is_hidden_dir(path):
    return os.path.isdir(path) and os.path.split(path)[1][0] == '.'


def is_tmp_dir(path):
    return os.path.isdir(path) and os.path.split(path)[1] == 'tmp'


def copy_app_to_remote(c):
    c.put(str(archive_path), str(app_target_dir))


def unzip_app_on_remote(c):
    remote_archive_path = app_target_dir / os.path.split(archive_path)[1]
    print(remote_archive_path)
    c.run(f'tar -xzvf {remote_archive_path}')
    c.run(f'rm {remote_archive_path}')


def build_docker_image(c):
    c.run(docker_build_cmd)


def run_docker_image(c):
    c.run(docker_run_cmd)
