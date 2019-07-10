import os
import tarfile

from fabric import task
from gitignore_parser import parse_gitignore
from pathlib import PurePosixPath, Path

user = 'dockeruser'
docker_image_name = 'jac241/muscle_segmentation'

docker_build_cmd = f'docker build -t {docker_image_name} .'

home_dir = PurePosixPath('/home') / user
dataset_dir = home_dir / 'data' / 'datasets'
output_dir = home_dir / 'output'
predictions_dir = home_dir / 'predictions'
training_output_dir = home_dir / 'training_output'

app_target_dir = home_dir / 'test'

docker_app_dir = PurePosixPath('/opt/app')
docker_dataset_dir = docker_app_dir / 'datasets'
docker_output_dir = docker_app_dir / 'output'
docker_predictions_dir = docker_app_dir / 'predictions'
docker_training_output_dir = docker_app_dir / 'training_output'

local_root = Path(os.getcwd())
archive_path = local_root / 'tmp' / 'repo.tar.gz'
local_training_output_dir = local_root / 'from_remote' / 'training_output'



"""
USAGE: 
fab --prompt-for-login-password -H dockeruser@10.1.32.31 run-training

"""
@task
def run_training(c):
    make_app_target_dir(c)
    zip_app_locally()
    copy_app_to_remote(c)
    unzip_app_on_remote(c)
    create_run_output_dir(c)
    build_docker_image(c)

    docker_run_output_dir = docker_training_output_dir / 'td'
    remote_run_output_dir = training_output_dir / 'td'

    # create_remote_training_output_dir(c, output_dir=docker_run_output_dir)
    run_docker_image_for_training(c, output_dir=docker_run_output_dir)

    retrieve_training_output(c, remote_dir=remote_run_output_dir, local_dir=local_training_output_dir)

    c.run(f'ls -la {app_target_dir}')


def make_app_target_dir(c):
    c.run(f'mkdir -p {app_target_dir}')


def zip_app_locally():
    with tarfile.open(archive_path, 'w:gz') as archive:
        archive.add('./', exclude=exclude_select_dirs)


def exclude_select_dirs(path):
    try:
        return is_hidden_dir(path) or is_in_gitignore(Path(os.getcwd(), path))
    except IndexError:
        return False


def is_hidden_dir(path):
    return os.path.isdir(path) and os.path.split(path)[1][0] == '.'


is_in_gitignore = parse_gitignore('.gitignore', base_dir=os.getcwd())


def copy_app_to_remote(c):
    c.put(str(archive_path), str(app_target_dir))


def unzip_app_on_remote(c):
    remote_archive_path = app_target_dir / os.path.split(archive_path)[1]
    print(remote_archive_path)
    c.run(f'tar -xzf {remote_archive_path} --directory {app_target_dir}')
    c.run(f'rm {remote_archive_path}')


def create_run_output_dir(c, run_output_dirname='td'):
    c.run(f'mkdir -p {training_output_dir / run_output_dirname}')


def build_docker_image(c):
    c.run(f'cd {app_target_dir} && ' + docker_build_cmd)


def run_docker_image_for_training(c, output_dir):
    model_output_path = output_dir / 'test.h5'
    c.run(docker_run_cmd(f'python run_training.py --model_output_path {model_output_path}'))


def docker_run_cmd(args):

    return (
        f'docker run -u $(id -u):$(id -g) --runtime=nvidia '
        f'-v {dataset_dir}:{docker_dataset_dir} '
        f'-v {output_dir}:{docker_output_dir} '
        f'-v {predictions_dir}:{docker_predictions_dir} '
        f'-v {training_output_dir}:{docker_training_output_dir} '
        f'-t {docker_image_name} '
        f'{args} '
    )


def retrieve_training_output(c, remote_dir, local_dir):
    remote_tar_file_path = training_output_dir / 'td.tar.gz'
    c.run(f'tar -czvf {remote_tar_file_path} -C {training_output_dir} td')

    ensure_dir_exists(local_training_output_dir)

    local_tarfile_path = local_training_output_dir / 'td.tar.gz'
    c.get(remote_tar_file_path, local_tarfile_path)

    with tarfile.open(local_tarfile_path, 'r:gz') as archive:
        archive.extractall(path=local_training_output_dir)

    os.remove(local_tarfile_path)


def ensure_dir_exists(path):
    try:
        os.mkdir(local_training_output_dir)
    except FileExistsError:
        pass
