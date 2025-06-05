# Using the 5G SA Testbed Folder: OAI-CN, MX-RAN, MX-UE, MX-RIC and MX-XAPP

<!-- [toc] -->

## System Requirements and Testing

Tested on Intel i9-10920X @ 3.50GHz.

**Minimum Recommended:**

* **CPU:** 12+ cores @ 3,8 GHz. **AVX-512 is a must-have.**
* **RAM:** 32 GB.
* **OS:** Modern Linux (e.g., Ubuntu 20.04).
* Docker & Docker Compose (latest stable).

Higher specs improve performance under load and with advanced monitoring.

## Directory Structure
This document provides information on how to use the contents of this directory
```bash
├── docker-compose.yaml
├── mx-conf/
├── oai-cn/
└── sqlite3/
```
## Important file and folder descriptions

### `docker-compose.yaml`

* **Description:** This is the primary Docker Compose file used to define and manage the multi-container Docker application within this scenario. It orchestrates the different services required for the 5G SA setup, such as the OAI Core Network (CN), any monitoring tools ( involving the data stored in the `sqlite3` folder), and the `mx` container.

* **Usage:**
    * **Starting the environment:** Navigate to this directory in your terminal and run:
        ```bash
        docker-compose up -d
        ```
        This command will start all the services defined in the `docker-compose.yaml` file in detached mode.
    * **Stopping the environment:** To stop all running containers, use:
        ```bash
        docker-compose down
        ```
    * **Viewing the status of services:** Check the status of the containers using:
        ```bash
        docker-compose ps
        ```
    * **Viewing logs of a specific service:** To see the logs of a container named `<service_name>` (defined in the `docker-compose.yaml`), use:
        ```bash
        docker-compose logs <service_name> -f
        ```
    * **Accessing a container's shell:** To get a shell inside a running container named `<service_name>`, use:
        ```bash
        docker-compose exec <service_name> bash
        ```
        or
        ```bash
        docker-compose exec <service_name> sh
        ```
### `sqlite3/data/`

* **Description:** This subfolder within `sqlite3/` is where the `mx-xapp` stores its monitoring data in SQLite database files.
* **File: `xapp_db`**
    * **Description:** This file, `xapp_db` (likely without a file extension, or potentially with `.db`), is the SQLite database file used by the `mx-xapp` to persist monitoring information.
    * **Usage:**
        * **Accessing the Database:** To read the data stored by the `mx-xapp`, you can use the `sqlite3` command-line tool. This tool allows you to interact directly with SQLite databases.
        * **Steps to Read `xapp_db`:**

            1.  **Ensure `sqlite3` is installed:** If you don't have the `sqlite3` command-line tool installed on your host system, you'll need to install it. The installation process varies depending on your operating system:
                * **Debian/Ubuntu:**
                    ```bash
                    sudo apt update
                    sudo apt install sqlite3
                    ```
                * **Fedora/CentOS/RHEL:**
                    ```bash
                    sudo dnf install sqlite
                    ```
                * **macOS:** If you have Homebrew installed:
                    ```bash
                    brew install sqlite3
                    ```

            2.  **Navigate to the `sqlite3/data` directory:** Open your terminal and change the current directory to the location of the `xapp_db` file within your project's `sqlite3/data` folder
                ```bash
                cd sqlite3/data
                ```

            3.  **Open the `xapp_db` database:** Use the `sqlite3` command followed by the filename:
                ```bash
                sqlite3 xapp_db
                ```

            4.  **Explore the database:** Once inside the SQLite shell, you can use various commands to inspect the database:
                * **.tables:** To list all the tables in the `xapp_db` database.
                    ```sqlite
                    .tables
                    ```
                * **.schema <table_name>:** To view the schema (structure) of a specific table. Replace `<table_name>` with the actual name of a table you found using `.tables`.
                    ```sqlite
                    .schema MAC_UE  -- Example: if a table is named 'MAC_UE'
                    ```
                * **.exit:** To exit the SQLite shell and return to your regular terminal prompt.
                    ```sqlite
                    .exit
                    ```

### `mx-conf/xapp.yaml` - `xapp_sub_cust_sm` Section

* **Description:** This section within the `xapp.yaml` file defines subscriptions to various Custom Service Models (SMs) for the `mx-xapp`. It allows you to specify which types of RAN information the xApp should receive and at what reporting frequency.

* **Key Configuration Parameters:**
    * `runtime_sec`: Sets the duration for which these subscriptions will be active. A value of `-1` typically means the subscriptions will run indefinitely.
    * Each numbered entry (e.g., `1:`, `2:`, `3:`) represents a distinct subscription to a Custom Service Model:
        * **`name`**: Specifies the name of the Custom Service Model to subscribe to (e.g., `mac`, `rlc`, `pdcp`).
        * **`periodicity_ms`**: Defines the reporting periodicity in milliseconds. This value determines how often the xApp will receive updates for the specified Service Model. **You can change this value** to adjust the frequency of the received monitoring data. Commonly supported values include `1`, `2`, `5`, `10`, `100`, or `1000` milliseconds.

* **Example Subscriptions:**
    ```yaml
    xapp_sub_cust_sm:
      runtime_sec: -1
      1:
        name: mac
        periodicity_ms: 10  # MAC layer information reported every 10 milliseconds
      2:
        name: rlc
        periodicity_ms: 100 # RLC layer information reported every 100 milliseconds
      3:
        name: pdcp
        periodicity_ms: 5   # PDCP layer information reported every 5 milliseconds
    ```

* **Supported Custom Service Models:** The `mx-xapp` and the underlying FlexRIC framework support various Custom Service Models, allowing you to monitor different aspects of the RAN. The example shows `mac`, `rlc`, and `pdcp`. **In addition to these, other supported Custom Service Models include:**
    * **`slice`**: For monitoring KPIs and information specific to network slices.
    * **`gtp`**: For monitoring data related to GTP (GPRS Tunneling Protocol) tunnels.

* **Adding or Modifying Subscriptions:** To subscribe to additional Custom Service Models or to change the reporting periodicity of existing ones, you can modify this `xapp_sub_cust_sm` section in the `xapp.yaml` file. Simply add a new numbered entry with the `name` of the desired Service Model and the `periodicity_ms` value. For example, to subscribe to the `slice` Service Model with a periodicity of 1000 milliseconds:

    ```yaml
    xapp_sub_cust_sm:
      runtime_sec: -1
      1:
        name: mac
        periodicity_ms: 10
      2:
        name: rlc
        periodicity_ms: 100
      3:
        name: pdcp
        periodicity_ms: 5
      4:
        name: slice
        periodicity_ms: 1000 # Slice-specific information reported every 1000 milliseconds
    ```

## Workflow

The typical workflow for using this folder likely involves:

1.  **Using the `docker-compose.yaml` file** to start, stop, and manage the entire 5G SA environment, including `mx-xapp` interacting with the SQLite database.
2.  **Capturing the SQLite database structure** within `sqlite3/` is as expected for the xApp's monitoring data.
3.  **Accessing the running containers** to further investigate configurations, logs, or to interact with the applications directly.

