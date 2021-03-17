import json
import subprocess

def daphne(args, cwd='../HW2/daphne'):
    # proc = subprocess.run(['/Users/berendz/bin/lein','run'] + args,
    #                       capture_output=True, cwd=cwd)
    proc = subprocess.run(['lein','run','-f','json'] + args,
                          stdout=subprocess.PIPE,stderr=subprocess.PIPE, cwd=cwd, shell=True)
    if(proc.returncode != 0):
        raise Exception(proc.stdout.decode() + proc.stderr.decode())
    return json.loads(proc.stdout)

