import subprocess
import json
import time
import os

encodings = ["atchley", "kidera"]
# encodings = ["aaprop"] #, "random"] 

for i in range(5):
    for encoding in encodings:
        
        config_path = f"config-{encoding}.json"
        with open(config_path, "r") as file:
            config = json.load(file)
        
        config["output-path"] = f"trained-{encoding}-{i}"
        
        with open(f"config-{encoding}-{i}.json", "w") as file:
            json.dump(config, file, indent=4)
            file.flush()
            os.fsync(file.fileno())
        
        # time.sleep(10)

        working_dir = "/cs/student/projects1/2020/cheuyuen/"
        log_file_name = f"{encoding}-{i}.log"
        
        # The command to be executed inside the screen session
        command = f'python trainer-symbolic.py -c config-{encoding}-{i}.json --log-file {log_file_name}\n'
        
        # Start a detached screen session with a specific name
        screen_start_command = f'screen -dmS {encoding}_{i}'
        subprocess.run(screen_start_command, shell=True, check=True)
        
        # Prepare the command to change directory, activate the virtual environment, and run the Python script
        full_command = f'cd {working_dir}; source tcr-cancer./bin/activate.csh; {command}'
        
        # Send the command to the screen session
        screen_stuff_command = f'screen -S {encoding}_{i} -X stuff "{full_command}"'
        subprocess.run(screen_stuff_command, shell=True, check=True)