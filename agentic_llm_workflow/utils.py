"""
This file provides a collection of utilities to manage the BubbleRAN environment,
including network lifecycle control (start, stop, status checks), traffic generation,
parameter changes and fetching in gNodeB configurations, and database synchronization between live
and persistent databases. It also supports initializing historical KPI data from CSV 
files and formatted output of runtime graph and node updates.

"""

from langchain_core.messages import convert_to_messages
import yaml
import sqlite3
import csv
import subprocess
import time
import os

def check_network_status(print_output=False):
    """
    Checks if all services defined in the BubbleRAN Docker Compose file are running.

    Args:
        print_output (bool): Whether to print status messages.

    Returns:
        bool: True if all services are running, False otherwise.
    """

    # Load network setup path from config.yaml
    bubbleran_network_setup_path = yaml.safe_load(open('config.yaml', 'r'))['bubbleran_network_setup']
    # Resolve the absolute path to the Docker Compose file
    # docker_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), bubbleran_network_setup_path))
    docker_file = os.path.join(bubbleran_network_setup_path, "docker-compose.yaml")
    try:
        with open(docker_file, 'r') as f:
            compose_data = yaml.safe_load(f)
        # Load services from the Docker Compose file
        expected_services = list(compose_data.get("services", {}).keys())
        result = subprocess.run(["docker", "ps", "--format", "{{.Names}}"], capture_output=True, text=True, check=True)
        # Get the names of currently running Docker containers
        running_containers = result.stdout.strip().splitlines()
        # Identify services that are expected but not running
        missing_containers = [service for service in expected_services if service not in running_containers]

        # Print the status report based on findings
        if missing_containers:
            if print_output:
                print(f"Running services: {running_containers}\nFailed services: ", missing_containers)
            return False
        else:
            if print_output:
                print("\nAll expected services are running successfully.")
            return True
        
    except Exception as e:
        print(f"Error while checking network status: {e}")
    return False


def start_network():
    """
    Starts the Docker-based BubbleRAN network if not already running.
    Yields:
        str: Status messages during the startup process.
    """
    # Check if Docker services are already running
    if check_network_status():
        yield "Dockers already running"
        return

    # Load network setup path from configuration file
    bubbleran_network_setup_path = "../" + yaml.safe_load(open('config.yaml', 'r'))['bubbleran_network_setup']
    # docker_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), bubbleran_network_setup_path))
    if 'BUBBLERAN_HOST_PWD' in os.environ:
        docker_dir = os.path.join(os.environ['BUBBLERAN_HOST_PWD'], yaml.safe_load(open('config.yaml', 'r'))['bubbleran_network_setup'])
    else:
        docker_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), bubbleran_network_setup_path))
    docker_run_command = ["docker", "compose", "up", "-d"] 

    try:
        # Attempt to start the Docker services
        yield("(1/3) Starting Docker containers... ")
        print("Starting Docker containers... ")
        subprocess.run(docker_run_command, cwd=docker_dir, check=True)
        

        # Check if additional UE connection setup is needed
        if "5g-sa-nr-sim" in bubbleran_network_setup_path:
            yield("(2/3) Connecting to User Equipment... ")
            print("Connecting to User Equipment... ")

            # Prepare command to start the iperf3 server inside the 'mx-ue' container
            start_server_command = (
                    "docker exec mx-ue sh -c "
                    "'LD_LIBRARY_PATH=/opt/hydra/usr/lib:/opt/hydra/usr/lib/x86_64-linux-gnu "
                    "/opt/hydra/usr/bin/iperf3 -s 10.0.0.2'"
                )
            subprocess.Popen(
                    start_server_command,
                    cwd=docker_dir,
                    shell=True,
                    # stdout=subprocess.DEVNULL,  #  Uncomment to suppress output
                    # stderr=subprocess.DEVNULL,  #  Uncomment to suppress errors
                )
            time.sleep(2)
            yield("(3/3) Initializing KPI database...")
            time.sleep(10)
            yield("(3/3) Almost done... just a few more seconds.")
            time.sleep(10)
        else:
            # To start UE connection for USRP-based setups, please refer to README
            pass
        
    except subprocess.CalledProcessError as e:
        print(f"Error: Failed to start Docker containers. Details: {e}")
        stop_network()
    except Exception as e:
        # Log unexpected errors and yield failure message
        print(f"Unexpected error occurred while starting network: {e}")
        yield(f"Stopping network due to error: {e}")
        stop_network()
    
    # Final status verification after attempting to start network
    if check_network_status():
        yield "Successfully started network"
    else:
        yield "Error: Network failed to start."
        stop_network()
    return
    

def stop_network(reset_db = False):
    """
    Stops the Docker-based BubbleRAN network and optionally resets local databases.
    
    Args:
        reset_db (bool): If True, clears database files after stopping the network.

    Yields:
        str: Status messages during the shutdown process.
    """
    # Load network setup path from configuration file
    bubbleran_network_setup_path = "../" + yaml.safe_load(open('config.yaml', 'r'))['bubbleran_network_setup']
    # docker_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), bubbleran_network_setup_path))

    if 'BUBBLERAN_HOST_PWD' in os.environ:
        docker_dir = os.path.join(os.environ['BUBBLERAN_HOST_PWD'], yaml.safe_load(open('config.yaml', 'r'))['bubbleran_network_setup'])
    else:
        docker_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), bubbleran_network_setup_path))

    # Command to stop Docker containers
    docker_stop_command = ["docker", "compose", "down"] 
    try:
        # Attempt to stop the Docker services
        print("Stopping Docker containers... ")
        yield("Stopping Docker containers... ")
        subprocess.run(docker_stop_command, cwd=docker_dir, check=True)
        time.sleep(4)
    except subprocess.CalledProcessError as e:
        print(f"Error: Failed to stop Docker containers. Details: {e}")
        yield(f"Error: Failed to stop Docker containers. Details: {e}")
        time.sleep(4)
    except Exception as e:
        print(f"Error: Unexpected error occurred while stopping network: {e}")
        yield(f"Error: Unexpected error occurred while stopping network: {e}")
        time.sleep(4)
    
    # Optionally reset database files if requested
    if reset_db==True:
        return "Success"
        reset_commands = [
            "sudo rm -rf 5g-sa-nr-sim/sqlite3/data",
            "rm ./persistent_db",
            "rm ./historical_db"
            ]
        for cmd in reset_commands:
            subprocess.run(cmd, shell=True, check=True)

    return "Success"


def add_traffic(traffic_bandwidth=100, traffic_time=15):
    """
    Starts traffic generation using iperf3 in the 'oai-ext-dn' BubbleRAN container.

    Args:
        traffic_bandwidth (int): Bandwidth in Mbps to generate. Default is 100 Mbps.
        traffic_time (int): Duration in seconds for traffic generation. Default is 15 seconds.

    Returns:
        str: Status message indicating success or failure.
    """

    # Load network setup path from configuration file
    bubbleran_network_setup_path = "../" + yaml.safe_load(open('config.yaml', 'r'))['bubbleran_network_setup']
    if 'BUBBLERAN_HOST_PWD' in os.environ:
        docker_dir = os.path.join(os.environ['BUBBLERAN_HOST_PWD'], yaml.safe_load(open('config.yaml', 'r'))['bubbleran_network_setup'])
    else:
        docker_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), bubbleran_network_setup_path))
    

    # Prepare iperf3 traffic generation command
    if "5g-sa-nr-sim" not in bubbleran_network_setup_path:
        return "Warning: Please follow the instructions in README for USRP traffic."
    add_traffic_command = f"docker exec oai-ext-dn iperf3 -B 192.168.70.135 -b {traffic_bandwidth}M -c 10.0.0.2 -t {traffic_time}"

    try:
        # Run the traffic generation command asynchronously to avoid blocking
        subprocess.Popen(
            add_traffic_command,
            cwd=docker_dir,
            shell=True,
            stdout=subprocess.DEVNULL,  #  Comment to suppress output
            # stderr=subprocess.DEVNULL,  #  Uncomment to suppress errors
        )
        print("Traffic generation started successfully.")
        return "Success: Traffic generation started successfully."
    except subprocess.CalledProcessError as e:
        print(f"Error: Failed to add traffic. Details: {e}")
    except Exception as e:
        print(f"Unexpected error while adding traffic: {e}")
    return "Error: Failed to start traffic generation."


def update_value_in_gnb(var, val):
    """
    Updates specified parameter in gNodeB configuration files.

    Args:
        var (str): The parameter to update (e.g., p0_nominal, att_tx, att_rx, dl_carrierBandwidth, ul_carrierBandwidth).
        val (any): The new value to set for the parameter.

    """
    
    # Handle direct scalar parameter updates in gNodeB config (e.g., p0_nominal, att_tx, att_rx)
    print(f"\n\n\nUPDATING {var} tp {val} in gnb utils")
    if var in ["p0_nominal", "att_tx", "att_rx"]:
        file_path = yaml.safe_load(open('config.yaml', 'r'))['bubbleran_network_setup'] + "/mx-conf/gnb.conf"
        with open(file_path, 'r') as file:
            lines = file.readlines()

        for i, line in enumerate(lines):
            if line.strip().startswith(var):
                # Replace the existing value while preserving formatting
                cur_num = line.split('=')[1].strip().strip(";")
                lines[i] = lines[i].replace(cur_num, str(val))
                break

        with open(file_path, 'w') as file:
            file.writelines(lines)
            print(f"Updated '{var}' to '{val}' in gNodeB configuration file: {file_path}")
    
    # Handle bandwidth profile updates (24, 51, 106) for specific carrierBandwidth USRP configurations
    elif var in ["dl_carrierBandwidth", "ul_carrierBandwidth"]:
        # Predefined bandwidth profiles with their corresponding parameters
        ul_dl_carrierBandwidth_dict = {}
        ul_dl_carrierBandwidth_dict["24"] = {"absoluteFrequencySSB": 640320,
                    "dl_absoluteFrequencyPointA": 640032, 
                    "dl_carrierBandwidth": 24, 
                    "ul_carrierBandwidth": 24,
                    "initialDLBWPlocationAndBandwidth": 6325,
                    "initialULBWPlocationAndBandwidth": 6325,
                    "initialDLBWPcontrolResourceSetZero": 2,
                    "sib1_tda": "uncomment",
                    "local_rf": "uncomment",
                    "imsi": "001010000000002",
                    "ssb": 24,
                    "nrb": 24,
                    "center_frequency_hz": "3604800000L",
                    }
        ul_dl_carrierBandwidth_dict["51"] = {"absoluteFrequencySSB": 640704,
                    "dl_absoluteFrequencyPointA": 639996, 
                    "dl_carrierBandwidth": 51, 
                    "ul_carrierBandwidth": 51,
                    "initialDLBWPlocationAndBandwidth": 13750,
                    "initialULBWPlocationAndBandwidth": 13750,
                    "initialDLBWPcontrolResourceSetZero": 12,
                    "sib1_tda": "comment",
                    "local_rf": "comment",
                    "imsi": "001010000000002",
                    "ssb": 234,
                    "nrb": 51,
                    "center_frequency_hz": "3609120000L",
                    }
        ul_dl_carrierBandwidth_dict["106"] = {"absoluteFrequencySSB": 641280,
                    "dl_absoluteFrequencyPointA": 640008, 
                    "dl_carrierBandwidth": 106, 
                    "ul_carrierBandwidth": 106,
                    "initialDLBWPlocationAndBandwidth": 28875,
                    "initialULBWPlocationAndBandwidth": 28875,
                    "initialDLBWPcontrolResourceSetZero": 12,
                    "sib1_tda": "comment",
                    "local_rf": "comment",
                    "imsi": "001010000000001",
                    "ssb": 516,
                    "nrb": 106,
                    "center_frequency_hz": "3619200000L",
                    }
        # Retrieve the profile based on the provided value
        current_ul_dl_carrierBandwidth_dict = ul_dl_carrierBandwidth_dict.get(str(val))
        if not current_ul_dl_carrierBandwidth_dict:
            print(f"Invalid bandwidth profile: {val}")
            return

        # Load network setup path from configuration file
        bubbleran_setup = yaml.safe_load(open('config.yaml', 'r'))['bubbleran_network_setup']

        # Target configuration files to update
        if "5g-sa-nr-sim" in bubbleran_setup:
            target_files = [
                os.path.join(bubbleran_setup, "mx-conf", "gnb.conf"),
                os.path.join(bubbleran_setup, "mx-conf", "nr-rfsim.conf")
            ]
        else:
            target_files = [
                os.path.join(bubbleran_setup, "mx-conf", "gnb.conf")
            ]            
        for file_path in target_files:
            with open(file_path, 'r') as file:
                lines = file.readlines()

            for i, line in enumerate(lines):
                cur_key = line.split("=")[0].strip()
                if cur_key in current_ul_dl_carrierBandwidth_dict.keys():

                    if current_ul_dl_carrierBandwidth_dict[cur_key] == "comment":
                        # Comment the line if instructed
                        tmp_line = lines[i].strip()
                        new_line = "# "+tmp_line
                        lines[i] = lines[i].replace(tmp_line, new_line)

                    elif current_ul_dl_carrierBandwidth_dict[cur_key] == "uncomment":
                        # Uncomment the line if instructed
                        tmp_line = lines[i].strip()
                        new_line = tmp_line.strip("#").strip()
                        lines[i] = lines[i].replace(tmp_line, new_line)

                    else:
                        # Replace the existing value with the new one
                        cur_num = line.split('=')[1].split(";")[0].strip()
                        lines[i] = lines[i].replace(cur_num, str(current_ul_dl_carrierBandwidth_dict[cur_key]))
                    
                    current_ul_dl_carrierBandwidth_dict[cur_key] = None

            with open(file_path, 'w') as file:
                file.writelines(lines)
                print(f"Updated '{var}' to '{val}' in configuration file: {file_path}")

    # Handle unsupported parameter cases
    else:
        print(f"Error: Unsupported parameter '{var}' for gNodeB configuration update.")


def find_value_in_gnb(var):
    """
    Retrieves the integer value of a specified parameter from the gNodeB configuration file.

    Args:
        var (str): The parameter name to search for.

    Returns:
        int: The integer value of the parameter if found; otherwise, returns None by default.
    """
    
    # Load the gNodeB configuration file path from config.yaml
    file_path = yaml.safe_load(open('config.yaml', 'r'))['bubbleran_network_setup'] + "/mx-conf/gnb.conf"

    val = None
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    # Search for the target parameter
    for i, line in enumerate(lines):
        if line.strip().startswith(var):
            # Extract and convert the value to integer
            val = int(line.split('=')[1].strip().strip(";"))
            break

    return val

def update_value_in_db(vars_dict):
    """
    Updates the persistent database with new rows from the live BubbleRAN database, 
    adding provided configuration values to each row.

    Args:
        vars_dict (dict): Dictionary containing the following keys:
                          - 'dl_carrierBandwidth'
                          - 'p0_nominal'
                          - 'ul_carrierBandwidth'
                          - 'att_tx'
                          - 'att_rx'

    """
    # Ensure the data directory exists for xapp_db
    bubbleran_network_setup_path = yaml.safe_load(open('config.yaml', 'r'))['bubbleran_network_setup']
    try:
        # Extract parameter values from the input dictionary
        dl_carrierBandwidth_val = vars_dict["dl_carrierBandwidth"]
        p0_nominal_val = vars_dict["p0_nominal"]
        ul_carrierBandwidth_val = vars_dict["ul_carrierBandwidth"]
        att_tx_val = vars_dict["att_tx"]
        att_rx_val = vars_dict["att_rx"]

        # Load database paths and table name from config
        live_db_path = bubbleran_network_setup_path + "/sqlite3/data/xapp_db"
        persistent_db_path = yaml.safe_load(open('config.yaml', 'r'))['persistent_db_path']

        # Connect to both live (temp) and persistent databases
        temp_conn = sqlite3.connect(live_db_path)
        temp_cursor = temp_conn.cursor()
        persistent_conn = sqlite3.connect(persistent_db_path)
        persistent_cursor = persistent_conn.cursor()

        if 'BUBBLERAN_HOST_PWD' in os.environ:
            os.system(f"chown $USER:$USER {live_db_path}")
            os.system(f"chmod 666 {live_db_path}")
            os.system(f"chown $USER:$USER {persistent_db_path}")
            os.system(f"chmod 666 {persistent_db_path}")
        else:
            # Update file permissions to ensure write access
            os.system(f"sudo chown $USER:$USER {live_db_path}")
            os.system(f"chmod 666 {live_db_path}")
            os.system(f"sudo chown $USER:$USER {persistent_db_path}")
            os.system(f"chmod 666 {persistent_db_path}")

        table_name = yaml.safe_load(open('config.yaml', 'r'))['table_name']
        

        # Check if MAC_UE table exists in persistent_db_path
        persistent_cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}'")
        table_exists = persistent_cursor.fetchone()

        if not table_exists:
            # Mirror the structure from the live database and add extra columns
            temp_cursor.execute(f"PRAGMA table_info({table_name})")
            columns = [col[1] + ' ' + col[2] for col in temp_cursor.fetchall()]
            columns.append("p0_nominal TEXT")
            columns.append("dl_carrierBandwidth TEXT")
            columns.append("ul_carrierBandwidth TEXT")
            columns.append("att_tx TEXT")
            columns.append("att_rx TEXT")
            create_query = f"CREATE TABLE {table_name} ({', '.join(columns)})"
            persistent_cursor.execute(create_query)
            persistent_conn.commit()
            print(f"Created {table_name} table in persistent database.")
        
        # Find the latest timestamp to avoid duplicate inserts
        persistent_cursor.execute(f"SELECT MAX(tstamp) FROM {table_name}")
        latest_tstamp = persistent_cursor.fetchone()[0] or 0

        # Fetch new rows from the live database with additional configuration values
        temp_cursor.execute(
                f"""SELECT *, ? AS p0_nominal, ? AS dl_carrierBandwidth, ? AS ul_carrierBandwidth, 
                            ? AS att_tx, ? AS att_rx 
                    FROM {table_name} 
                    WHERE tstamp > ?""",
                (p0_nominal_val, dl_carrierBandwidth_val, ul_carrierBandwidth_val, att_tx_val, att_rx_val, latest_tstamp)
            )
        new_rows = temp_cursor.fetchall()

        if new_rows:
            # Prepare column names for the insert operation
            temp_cursor.execute(f"PRAGMA table_info({table_name})")
            columns = [col[1] for col in temp_cursor.fetchall()] + [
                        "p0_nominal", "dl_carrierBandwidth", "ul_carrierBandwidth", "att_tx", "att_rx"
                    ]
            placeholders = ', '.join(['?'] * len(columns))

            # Insert the new rows into the persistent database
            insert_query = f"INSERT INTO {table_name} ({', '.join(columns)}) VALUES ({placeholders})"
            persistent_cursor.executemany(insert_query, new_rows)
            persistent_conn.commit()
            print(f"Inserted {len(new_rows)} new rows into '{table_name}'.")
        else:
            print("No new data to insert. Persistent database is up-to-date.")

        # Close both database connections
        temp_conn.close()
        persistent_conn.close()

    except Exception as e:
        error_message = f"Error: Unexpected error encountered while updating persistent database: {e}"
        print(error_message)
        return error_message


def read_historical_data():
    """
    Initializes a SQLite database 'historical_db' from 'historical_data.csv' if it does not already exist.
    Creates a 'kpis' table and populates it with data from the CSV file, applying data transformations where needed.
    """

    historical_db_path = yaml.safe_load(open('config.yaml', 'r'))['historical_db_path']
    table_name = "kpis"
    csv_filename = "data/historical_data.csv"

    # Skip initialization if the database already exists
    if os.path.exists(historical_db_path):
        return

    try:
        # Connect to the SQLite database (creates it if not present)
        conn = sqlite3.connect(historical_db_path)
        cursor = conn.cursor()

        # Create the table if it does not already exist
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                Parameter TEXT,
                Value INT,
                name TEXT,
                mcs_DL FLOAT,
                retx FLOAT,
                txo FLOAT,
                bitrate_DL FLOAT,
                snr FLOAT,
                mcs_UL FLOAT,
                rxko_gNB FLOAT,
                rxok FLOAT,
                bitrate_UL FLOAT,
                LDPC_iterations FLOAT
            )
        """)

        # Open the CSV file and begin reading data
        with open(csv_filename, newline='') as csv_file:
            reader = csv.DictReader(csv_file)

            for row in reader:
                value = row["Value"]
                # Handle special case for 'p0 nominal' by negating the value
                if row["Parameter"].strip().lower() == "p0 nominal":
                    try:
                        value = str(-float(value))
                    except ValueError:
                        print(f"Warning: Could not convert 'Value' to float for negation: {value}")

                # Remap 'tx_gain' to 'att_tx' with adjusted scaling
                if row["Parameter"] == "tx_gain":
                    row["Parameter"] = "att_tx"
                    value = 90-int(value)

                # Remap 'rx_gain' to 'att_rx' with adjusted scaling
                if row["Parameter"] == "rx_gain":
                    row["Parameter"] = "att_rx"
                    value = 70-int(value)
                    
                # Insert the processed row into the database
                cursor.execute(f"""
                    INSERT INTO {table_name} (
                        Parameter, Value, name, mcs_DL, retx, txo,
                        bitrate_DL, snr, mcs_UL, rxko_gNB, rxok, bitrate_UL, LDPC_iterations
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    row["Parameter"],
                    value,
                    row["name"],
                    row["mcs ( in DL)"],
                    float(row["retx (# of retransmissions requested by the UE)"]),
                    float(row["txok (# of successful transmissions)"]),
                    float(row['bit rate (average bitrate (at the MAC layer), in bits per second in DL)']),
                    float(row["snr"]),
                    row["mcs (UL)"],
                    float(row["rxko (# of retransmissions requested by the gNB)"]),
                    float(row["rxok (# of received uplink transport blocks without error)"]),
                    float(row["bitrate UL(mbps)"]),
                    row["# of iterations of the†LDPC decoder"]
                ))
        # Commit all changes and close the connection
        conn.commit()
        conn.close()
        print(f"Successfully created database '{historical_db_path}' and populated it with historical KPI data.")

    except Exception as e:
        # Clean up if database creation fails
        if os.path.exists(historical_db_path):
            os.remove(historical_db_path)
            print(f"Database '{historical_db_path}' removed due to error.")
        print(f"Error: Failed to create database '{historical_db_path}'")


def pretty_print_messages(update):
    """
    Pretty prints updates from graph or node messages.

    Args:
        update (dict or tuple): Contains the update information, optionally with namespace.
    """
    # Handle case where update includes namespace context
    if isinstance(update, tuple):
        ns, update = update

        # Skip printing for top-level graph updates without namespace details
        if len(ns) == 0:
            return

        graph_id = ns[-1].split(":")[0]
        print(f"--- Update received from subgraph: {graph_id} ---")

    # Iterate over each node and print its updates
    for node_name, node_update in update.items():
        print(f"--- Update received from node: {node_name} ---")

        # Process and pretty print each message
        for m in convert_to_messages(node_update["messages"]):
            m.pretty_print()

        print("-" * 50)  # Separator for readability




