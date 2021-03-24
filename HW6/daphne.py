import json
import subprocess


def daphne(args, cwd='../daphne'):
    proc = subprocess.run(['lein','run'] + args,
                          stdout=subprocess.PIPE,stderr=subprocess.PIPE, cwd=cwd, shell=True)
    if(proc.returncode != 0):
        raise Exception(proc.stdout.decode() + proc.stderr.decode())
    return json.loads(proc.stdout)
