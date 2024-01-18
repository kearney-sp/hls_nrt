from dask.distributed import LocalCluster, Client
import dask
from hlsstack.hls_funcs import fetch
from importlib.metadata import version



def launch_dask(cluster_loc='local', 
                num_processes=8,
                num_threads_per_processes=2, 
                mem_gb_per=2.5,
                num_jobs=16,
                partition='scavenger', 
                slurm_opts={'interface': 'ens7f0'},
                duration='02:00:00',
                wait_for_workers=True,
                wait_proportion=0.5,
                wait_timeout=120):
    if cluster_loc == 'local':
        print('   setting up Local cluster...')
        aws=False
        fetch.setup_env(aws=aws)
        cluster = LocalCluster(n_workers=num_processes,
                               threads_per_worker=num_threads_per_processes)
        client = Client(cluster)
        display(client)
    elif cluster_loc == 'hpc':
        import dask_jobqueue as jq
        djq_version = float('.'.join(version('dask_jobqueue').split('.')[:-1]))
        print('   setting up cluster on HPC...')
        aws=False
        fetch.setup_env(aws=aws)
        num_processes = 4
        num_threads_per_processes = 2
        mem = mem_gb_per*num_processes*num_threads_per_processes
        n_cores_per_job = num_processes*num_threads_per_processes
        if djq_version >= 0.8:
            clust = jq.SLURMCluster(queue=partition,
                                    processes=num_processes,
                                    cores=n_cores_per_job,
                                    memory=str(mem)+'GB',
                                    #interface='ib0',
                                    #interface='ens7f0',
                                    scheduler_options=slurm_opts,
                                    local_directory='$TMPDIR',
                                    death_timeout=120,
                                    walltime=duration,
                                    job_extra_directives=["--output=/dev/null","--error=/dev/null"])
        else:
            clust = jq.SLURMCluster(queue=partition,
                                    processes=num_processes,
                                    cores=n_cores_per_job,
                                    memory=str(mem)+'GB',
                                    #interface='ib0',
                                    #interface='ens7f0',
                                    scheduler_options=slurm_opts,
                                    local_directory='$TMPDIR',
                                    death_timeout=120,
                                    walltime=duration,
                                    job_extra=["--output=/dev/null","--error=/dev/null"])
            
        client=Client(clust)
        #Scale Cluster 
        clust.scale(jobs=num_jobs)
        if wait_for_workers:
            try:
                client.wait_for_workers(n_workers=num_jobs*num_processes*wait_proportion, timeout=wait_timeout)
            except dask.distributed.TimeoutError as e:
                print(str(num_jobs*num_processes) + ' workers may not be available. Displaying available workers.')
                #print(e)
                pass
    return client