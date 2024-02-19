# %%

# ==================================================================================================
# --- Imports
# ==================================================================================================
import copy
import os
import subprocess
import time
from pathlib import Path

import psutil
import tree_maker
import yaml


# ==================================================================================================
# --- Class for job submission
# ==================================================================================================
class ClusterSubmission:
    def __init__(self, config, path_root):
        # Configuration of the current generation
        self.config = config
        if config["run_on"] in ["local_pc", "htc", "slurm", "htc_docker", "slurm_docker"]:
            self.run_on = self.config["run_on"]
        else:
            raise ("Error: Submission mode specified is not yet implemented")

        # GPU configuration (for HTC)
        if config["context"] in ["cupy", "opencl"] and self.run_on in [
            "htc",
            "htc_docker",
            "slurm",
            "slurm_docker",
        ]:
            self.request_GPUs = 1
            self.slurm_queue_statement = ""
        else:
            self.request_GPUs = 0
            self.slurm_queue_statement = "#SBATCH --partition=slurm_hpc_acc"

        # Path to store the association between job path and job id after submission
        self.path_root = path_root
        self.path_dic_id_to_job = f"{self.path_root}/id_job.yaml"

        # Path to singularity image
        if "singularity_image" in self.config:
            self.path_image = self.config["singularity_image"]
        else:
            if self.run_on in ["slurm_docker", "htc_docker"]:
                raise ("Error: Singularity image must be specified for docker submission")
            else:
                # Needs to be defined, but irrelevant for local_pc, slurm and htc
                self.path_image = ""

        # Dictionnary of instructions for submission files
        self.dic_submission = {
            "local_pc": {
                "head": "# Running on local pc\n",
                "body": lambda path_node: f"bash {path_node}/run.sh &\n",
                "tail": f"#{self.run_on}\n",
                "submit_command": lambda filename: f"bash {filename}",
            },
            "slurm": {
                "head": "# Running on SLURM \n",
                "body": lambda path_node: (
                    f"sbatch --ntasks=2 "
                    f"{self.slurm_queue_statement.split(' ')[1] if len(self.slurm_queue_statement)>0 else self.slurm_queue_statement}"
                    f" --output=output.txt"
                    f" --error=error.txt --gres=gpu:{self.request_GPUs}"
                    f" {path_node}/run.sh\n"
                ),
                "tail": f"#{self.run_on}\n",
                "submit_command": lambda filename: f"bash {filename}",
            },
            "slurm_docker": {
                "head": lambda path_node: (
                    "#!/bin/bash\n"
                    + "# This is a SLURM submission file using Docker\n"
                    + self.slurm_queue_statement
                    + "\n"
                    + f"#SBATCH --output={path_node}/output.txt\n"
                    + f"#SBATCH --error={path_node}/error.txt\n"
                    + "#SBATCH --ntasks=2\n"
                    + f"#SBATCH --gres=gpu:{self.request_GPUs}\n"
                ),
                "body": lambda path_node: (
                    f"singularity exec {self.path_image} {path_node}/run.sh\n"
                ),
                "tail": f"#{self.run_on}\n",
                "submit_command": lambda filename: f"sbatch {filename}",
            },
            "htc": {
                "head": (
                    "# This is a HTCondor submission file\n"
                    + "error  = error.txt\n"
                    + "output = output.txt\n"
                    + "log  = log.txt\n"
                ),
                "body": (
                    lambda path_node: f"initialdir = {path_node}\n"
                    + f"executable = {path_node}/run.sh\n"
                    + f"request_GPUs = {self.request_GPUs}\n"
                    + "queue\n"
                ),
                "tail": f"#{self.run_on}\n",
                "submit_command": lambda filename: f"condor_submit {filename}",
            },
            "htc_docker": {
                "head": (
                    "# This is a HTCondor submission file using Docker\n"
                    + "error  = error.txt\n"
                    + "output = output.txt\n"
                    + "log  = log.txt\n"
                    + "universe = vanilla\n"
                    + "+SingularityImage ="
                    + f' "{self.path_image}"\n'
                ),
                "body": (
                    lambda path_node: f"initialdir = {path_node}\n"
                    + f"executable = {path_node}/run.sh\n"
                    + f"request_GPUs = {self.request_GPUs}\n"
                    + "queue\n"
                ),
                "tail": f"#{self.run_on}\n",
                "submit_command": lambda filename: f"condor_submit {filename}",
            },
        }

    # Getter for dic_id_to_job
    @property
    def dic_id_to_job(self):
        if os.path.isfile(self.path_dic_id_to_job):
            with open(self.path_dic_id_to_job, "r") as fid:
                return yaml.load(fid, Loader=yaml.FullLoader)
        else:
            return None

    # Setter for dic_id_to_job
    @dic_id_to_job.setter
    def dic_id_to_job(self, dic_id_to_job):
        assert isinstance(dic_id_to_job, dict)
        # Ensure all ids are integers
        dic_id_to_job = {int(id_job): job for id_job, job in dic_id_to_job.items()}

        # Write on disk
        with open(self.path_dic_id_to_job, "w") as fid:
            yaml.dump(dic_id_to_job, fid)

        # Wait 0.5s to make sure the file is written on disk
        time.sleep(0.5)

    def _update_dic_id_to_job(self, running_jobs, queuing_jobs):
        # Look for jobs in the dictionnary that are not running or queuing anymore
        dic_id_to_job = self.dic_id_to_job
        set_current_jobs = set(running_jobs + queuing_jobs)
        if dic_id_to_job is not None:
            for id_job, job in self.dic_id_to_job.items():
                if job not in set_current_jobs:
                    del dic_id_to_job[id_job]

            # Update (write on disk) dic_id_to_job
            self.dic_id_to_job = dic_id_to_job

    def _get_state_jobs(self, dic_id_to_job=None, verbose=True):
        if dic_id_to_job is None:
            dic_id_to_job = self.dic_id_to_job
        running_jobs = self.querying_jobs(dic_id_to_job=dic_id_to_job, status="running")
        queuing_jobs = self.querying_jobs(dic_id_to_job=dic_id_to_job, status="queuing")
        self._update_dic_id_to_job(running_jobs, queuing_jobs)
        if verbose:
            print(f"Running: \n" + "\n".join(running_jobs))
            print(f"queuing: \n" + "\n".join(queuing_jobs))
        return running_jobs, queuing_jobs

    @staticmethod
    def _get_path_job(path_node):
        path_job = copy.copy(path_node)

        # Fix for path
        if not path_node.endswith("/"):
            path_job += "/"

        # Only get path after master_study
        path_job = path_job.split("master_study")[1]

        return path_job

    @staticmethod
    def _test_node(node, path_job, running_jobs, queuing_jobs):
        # Test if node is running, queuing or completed
        if node.has_been("completed"):
            print(f"{path_job} is already completed.")
        elif path_job in running_jobs:
            print(f"{path_job} is already running.")
        elif path_job in queuing_jobs:
            print(f"{path_job} is already queuing.")
        else:
            return True
        return False

    def _write_sub_files_slurm(self, filename, running_jobs, queuing_jobs, list_of_nodes):
        l_filenames = []
        l_path_jobs = []
        for idx_node, node in enumerate(list_of_nodes):
            # Get path node
            path_node = node.get_abs_path()

            # Get corresponding path job
            path_job = self._get_path_job(path_node)

            # Test if node is running, queuing or completed
            if self._test_node(node, path_job, running_jobs, queuing_jobs):
                filename_node = f"{filename.split('.sub')[0]}_{idx_node}.sub"

                # Write the submission files
                print('Writing submission file for node "' + path_node + '"')
                with open(filename_node, "w") as fid:
                    # Careful, I implemented a fix for path due to the temporary home recovery folder
                    to_replace = "/storage-hpc/gpfs_data/HPC/home_recovery"
                    replacement = "/home/HPC"
                    fixed_path = path_node.replace(to_replace, replacement)
                    # update path for sed
                    to_replace = to_replace.replace("/", "\/")
                    replacement = replacement.replace("/", "\/")

                    # Head
                    fid.write(self.dic_submission[self.run_on]["head"](fixed_path))

                    # Mutate path in run.sh and other potentially problematic files
                    fid.write(f"sed -i 's/{to_replace}/{replacement}/' {fixed_path}/run.sh\n")
                    fid.write(f"sed -i 's/{to_replace}/{replacement}/' {fixed_path}/config.yaml\n")

                    # Body
                    fid.write(self.dic_submission[self.run_on]["body"](fixed_path))

                    # Tail
                    fid.write(self.dic_submission[self.run_on]["tail"])

                l_filenames.append(filename_node)
                l_path_jobs.append(path_job)
        return l_filenames, l_path_jobs

    def _write_sub_file(
        self, filename, running_jobs, queuing_jobs, list_of_nodes, write_htc_job_flavour=False
    ):
        # Get submission instructions
        str_head = self.dic_submission[self.run_on]["head"]
        str_body = self.dic_submission[self.run_on]["body"]
        str_tail = self.dic_submission[self.run_on]["tail"]

        # Flag to know if the file can be submitted (at least one job in it)
        ok_to_submit = False

        # Record list of jobs
        l_path_jobs = []

        # Write the submission file
        with open(filename, "w") as fid:
            fid.write(str_head)
            for node in list_of_nodes:
                # Get path node
                path_node = node.get_abs_path()

                # Get corresponding path job
                path_job = self._get_path_job(path_node)

                # Test if node is running, queuing or completed
                if self._test_node(node, path_job, running_jobs, queuing_jobs):
                    print('Writing submission command for node "' + path_node + '"')
                    # Write instruction for submission
                    fid.write(str_body(path_node))

                    # if user has defined a htc_job_flavor in config.yaml otherwise default is "espresso"
                    if write_htc_job_flavour:
                        if "htc_job_flavor" in self.config:
                            htc_job_flavor = self.config["htc_job_flavor"]
                        else:
                            print(
                                "Warning: htc_job_flavor not defined in config.yaml. Using espresso"
                                " as default"
                            )
                            htc_job_flavor = "espresso"
                        fid.write(f'+JobFlavour  = "{htc_job_flavor}"\n')

                    # Add job to list
                    l_path_jobs.append(path_job)

                    # Flag file
                    ok_to_submit = True

            # Tail instruction
            fid.write(str_tail)

        if not ok_to_submit:
            os.remove(filename)

        return ([filename], l_path_jobs) if ok_to_submit else ([], [])

    def _write_sub_files(self, filename, running_jobs, queuing_jobs, list_of_nodes):
        # Slurm docker is a peculiar case as one submission file must be created per job
        if self.run_on == "slurm_docker":
            return self._write_sub_files_slurm(filename, running_jobs, queuing_jobs, list_of_nodes)

        # htcondor, local_pc, etc.
        else:
            return self._write_sub_file(
                filename,
                running_jobs,
                queuing_jobs,
                list_of_nodes,
                write_htc_job_flavour=True if self.run_on in ["htc", "htc_docker"] else False,
            )

    def write_sub_files(self, list_of_nodes, filename="file.sub"):
        running_jobs, queuing_jobs = self._get_state_jobs(verbose=False)
        l_filenames, l_path_jobs = self._write_sub_files(
            filename, running_jobs, queuing_jobs, list_of_nodes
        )
        return l_filenames, l_path_jobs

    def submit(self, l_filenames, l_jobs):
        # Check that the submission file(s) is/are appropriate for the submission mode
        if len(l_filenames) > 1 and self.run_on != "slurm_docker":
            raise (
                "Error: Multiple submission files should not be implemented for this submission"
                " mode"
            )
        if len(l_filenames) == 0:
            print("No job being submitted.")

        # Submit
        dic_id_to_job_temp = {}
        idx_submission = 0
        for filename in l_filenames:
            if self.run_on in self.dic_submission:
                if self.run_on == "local_pc":
                    os.system(self.dic_submission[self.run_on]["submit_command"](filename))
                else:
                    process = subprocess.run(
                        self.dic_submission[self.run_on]["submit_command"](filename).split(" "),
                        capture_output=True,
                    )
                    output = process.stdout.decode("utf-8")
                    output_error = process.stderr.decode("utf-8")
                    if "ERROR" in output_error:
                        raise RuntimeError(f"Error in submission: {output}")
                    for line in output.split("\n"):
                        if "htc" in self.run_on:
                            if "cluster" in line:
                                cluster_id = int(line.split("cluster ")[1][:-1])
                                dic_id_to_job_temp[cluster_id] = l_jobs[idx_submission]
                                idx_submission += 1
                        elif "slurm" in self.run_on:
                            if "Submitted" in line:
                                job_id = int(line.split(" ")[3])
                                dic_id_to_job_temp[job_id] = l_jobs[idx_submission]
                                idx_submission += 1
            else:
                raise (f"Error: Submission mode {self.run_on} is not yet implemented")

        # Update and write the id-job file
        if len(dic_id_to_job_temp) > 0:
            assert len(dic_id_to_job_temp) == len(l_jobs)

        # Merge with the previous id-job file
        dic_id_to_job = self.dic_id_to_job

        # Update and write on disk
        if dic_id_to_job is not None:
            dic_id_to_job.update(dic_id_to_job_temp)
            self.dic_id_to_job = dic_id_to_job
        else:
            if len(dic_id_to_job_temp) > 0:
                dic_id_to_job = dic_id_to_job_temp
                self.dic_id_to_job = dic_id_to_job

        print("Jobs status after submission:")
        running_jobs, queuing_jobs = self._get_state_jobs(dic_id_to_job=dic_id_to_job, verbose=True)

    @staticmethod
    def _get_local_jobs():
        l_jobs = []
        # Warning, does not work at the moment in lxplus...
        for ps in psutil.pids():
            try:
                aux = psutil.Process(ps).cmdline()
            except:
                aux = []
            if len(aux) > 1:
                if "run.sh" in aux[-1]:
                    job = str(Path(aux[-1]).parent)

                    # Only get path after master_study
                    job = job.split("master_study")[1]

                    l_jobs.append(job + "/")
        return l_jobs

    @staticmethod
    def _get_condor_jobs(status, dic_id_to_job=None, force_query_individually=False):
        l_jobs = []
        dic_status = {"running": 1, "queuing": 2}
        condor_output = subprocess.run(["condor_q"], capture_output=True).stdout.decode("utf-8")

        # Check which jobs are running
        first_line = True
        first_missing_job = True
        for line in condor_output.split("\n")[4:]:
            if line == "":
                break
            jobid = int(line.split("ID:")[1][1:8])
            condor_status = line.split("      ")[1:5]  # Done, Run, Idle, and potentially Hold

            # If job is running/queuing, get the path to the job
            if condor_status[dic_status[status]] == "1":
                # Get path from dic_id_to_job if available
                if dic_id_to_job is not None:
                    if jobid in dic_id_to_job:
                        job = dic_id_to_job[jobid]
                        l_jobs.append(job)
                    else:
                        if first_missing_job:
                            print(
                                "Warning, some jobs are queuing/running and are not in the id-job"
                                " file. They may come from another study. Ignoring them."
                            )
                            first_missing_job = False

                # Query job individually if needed
                elif force_query_individually:
                    if first_line:
                        print(
                            "Warning, some jobs are queuing/running and the id-job file is"
                            " missing... Querying them individually."
                        )
                        first_line = False
                    job_details = subprocess.run(
                        ["condor_q", "-l", f"{jobid}"], capture_output=True
                    ).stdout.decode("utf-8")
                    job = job_details.split('Cmd = "')[1].split("run.sh")[0]

                    # Only get path after master_study
                    job = job.split("master_study")[1]
                    l_jobs.append(job)

                # Just ignore jobs
                else:
                    if first_line:
                        print(
                            "Warning, some jobs are queuing/running and the id-job file is"
                            " missing... Ignoring them."
                        )
                        first_line = False

        return l_jobs

    @staticmethod
    def _get_slurm_jobs(status, dic_id_to_job=None, force_query_individually=False):
        l_jobs = []
        dic_status = {"running": "RUNNING", "queuing": "PENDING"}
        username = (
            subprocess.run(["id", "-u", "-n"], capture_output=True).stdout.decode("utf-8").strip()
        )
        slurm_output = subprocess.run(
            ["squeue", "-u", f"{username}", "-t", dic_status[status]], capture_output=True
        ).stdout.decode("utf-8")

        # Get job id and details
        first_line = True
        first_missing_job = True
        for line in slurm_output.split("\n")[1:]:
            l_split = line.split()
            if len(l_split) == 0:
                break
            jobid = int(l_split[0])
            slurm_status = l_split[4]  # R or PD

            # Get path from dic_id_to_job if available
            if dic_id_to_job is not None:
                if jobid in dic_id_to_job:
                    job = dic_id_to_job[jobid]
                    l_jobs.append(job)
                else:
                    if first_missing_job:
                        print(
                            "Warning, some jobs are queuing/running and are not in the id-job"
                            " file. They may come from another study. Ignoring them."
                        )
                        first_missing_job = False

            # Else, query job individually
            elif force_query_individually:
                if first_line:
                    print(
                        "Warning, some jobs are queuing/running and the id-job file is"
                        " missing... Querying them individually."
                    )
                    first_line = False
                job_details = subprocess.run(
                    ["scontrol", "show", "jobid", "-dd", f"{jobid}"], capture_output=True
                ).stdout.decode("utf-8")
                if "run.sh" in job_details:
                    job = job_details.split("Command=")[1].split("run.sh")[0]
                else:
                    job = job_details.split("StdOut=")[1].split("output.txt")[0]

                # Only get path after master_study
                job = job.split("master_study")[1]
                l_jobs.append(job)

            # Just ignore jobs
            else:
                if first_line:
                    print(
                        "Warning, some jobs are queuing/running and the id-job file is"
                        " missing... Ignoring them."
                    )
                    first_line = False

        return l_jobs

    def querying_jobs(self, status="running", dic_id_to_job=None):
        l_jobs = []

        if self.run_on == "local_pc":
            if status == "running":
                l_jobs = self._get_local_jobs()
            else:
                # Always empty return as there is no queuing in local pc
                pass

        elif self.run_on == "htc" or self.run_on == "htc_docker":
            l_jobs = self._get_condor_jobs(status, dic_id_to_job)

        elif self.run_on == "slurm" or self.run_on == "slurm_docker":
            l_jobs = self._get_slurm_jobs(status, dic_id_to_job)

        else:
            print("Querying jobs are not implemented yet for this submission mode")
        return l_jobs


# ==================================================================================================
# --- Main submission function
# ==================================================================================================
def submit_jobs_generation(root, generation=1):
    # Define a dictionnary that associates a name to each generation number
    dic_int_to_str = {1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth"}
    if generation not in dic_int_to_str:
        raise (f"Error: Generation {generation} is not implemented")

    # Submit all the pending jobs of a given generation
    config_generation = root.parameters["generations"][f"{generation}"]
    cluster_submission = ClusterSubmission(config_generation, root.get_abs_path())
    path_file = f"submission_files/{dic_int_to_str[generation]}_generation.sub"
    l_filenames, l_path_jobs = cluster_submission.write_sub_files(
        root.generation(generation), path_file
    )
    cluster_submission.submit(l_filenames, l_path_jobs)


def submit_jobs(study_name, print_uncompleted_jobs=False):
    # Add suffix to the root node path to handle scans that are not in the root directory
    fix = "/scans/" + study_name
    root = tree_maker.tree_from_json(fix[1:] + "/tree_maker.json")
    root.add_suffix(suffix=fix)

    # Check that the study is not done yet
    if root.has_been("completed"):
        print("All descendants of root are completed!")
    else:
        # Check generation 1
        gen_1_completed = all([node.has_been("completed") for node in root.generation(1)])
        if not gen_1_completed:
            print("######## Taking care of generation 1 ########")
            submit_jobs_generation(root, generation=1)
        else:
            print("Generation 1 is already completed.")

        # Check generation 2
        gen_2_completed = all([node.has_been("completed") for node in root.generation(2)])
        if gen_1_completed and not gen_2_completed:
            print("######## Taking care of generation 2 ########")
            submit_jobs_generation(root, generation=2)
        else:
            if not gen_1_completed:
                pass
            else:
                print("Generation 2 is already completed.")

        # We assume there's no generation 3
        if all([descendant.has_been("completed") for descendant in root.descendants]):
            root.tag_as("completed")
            print("All descendants of root are completed!")

        # Print remaining jobs
        if print_uncompleted_jobs:
            for descendant in root.descendants:
                if descendant.has_not_been("completed"):
                    print("To be completed: " + descendant.get_abs_path())


# ==================================================================================================
# --- Submission
# ==================================================================================================
# Load the tree from a yaml and submit the jobs that haven't been completed yet
if __name__ == "__main__":
    # Define study
    study_name = "example_tunescan"

    # Submit jobs
    submit_jobs(study_name)

# %%
