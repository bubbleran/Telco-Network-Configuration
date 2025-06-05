"""
LangGraph Agent Tools for BubbleRAN Orchestration

This file defines a collection of tools used by LangGraph agents to interact with 
the BubbleRAN environment. These tools enable the agents to perform database queries, 
extract or update gNodeB configuration parameters, and perform weighted analysis 
on KPI datasets to make data-driven decisions to optimize network behavior.
"""

from langchain_nvidia_ai_endpoints import ChatNVIDIA
from typing import Literal, Union, Annotated, Optional
from typing_extensions import TypedDict
from langgraph.types import Command
from langgraph.prebuilt import create_react_agent
from langgraph.graph import StateGraph, START, END, MessagesState
from langchain_core.messages import AnyMessage, convert_to_messages
from langchain_core.tools import tool
import pandas as pd
import sqlite3
from io import StringIO
import json
import yaml
import subprocess


@tool
def execute_xapp_sql(sql_query: Annotated[str, "The SQL Query to be executed on the database"]):
    """
    Executes the provided SQL query on the persistent database 
    and returns the result as a formatted string.

    Args:
        sql_query (str): The SQL query to execute.

    Returns:
        str: The query result formatted as a table string.
    """
    # Load the database path from config.yaml (uses persistent database)
    db_path = yaml.safe_load(open('config.yaml', 'r'))['persistent_db_path']

    # Connect to the SQLite database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Execute the provided SQL query
    cursor.execute(sql_query)
    # print("SQL query execution completed.")

    # Fetch all rows and extract column names
    rows = cursor.fetchall()
    columns = [desc[0] for desc in cursor.description]  

    # Format the results into a pandas DataFrame
    result_df = pd.DataFrame(rows, columns=columns)
    conn.close()  

    # Convert DataFrame to a formatted string without index numbers
    result_df_string = result_df.to_string(index=False)
    return result_df_string


@tool
def execute_historical_sql(sql_query: Annotated[str, "The SQL Query to be executed on the database"]):
    """
    Executes the provided SQL query on the historical KPI database 
    and returns the result as a formatted string.

    Args:
        sql_query (str): The SQL query to execute.

    Returns:
        str: The query result formatted as a table string.
    """

    # Load the database path from config.yaml (uses historical database)
    db_path = yaml.safe_load(open('config.yaml', 'r'))['historical_db_path']

    # Connect to the SQLite database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Execute the provided SQL query
    cursor.execute(sql_query)
    # print("Historical SQL query executed successfully.")

    # Fetch all results and extract column names
    rows = cursor.fetchall()
    columns = [desc[0] for desc in cursor.description] 

    # Convert results to a pandas DataFrame
    result_df = pd.DataFrame(rows, columns=columns)
    conn.close()  

    # Format DataFrame as a string without index numbers
    result_df_string = result_df.to_string(index=False)
    return result_df_string

@tool
def find_value_in_gnb(var: Annotated[str, "Variable to search for in the GNB config file."]) -> int:
    """
    Retrieves the integer value of a specified parameter from the gNodeB configuration file.

    Args:
        var (str): The name of the parameter to retrieve (e.g., 'p0_nominal', 'dl_carrierBandwidth').

    Returns:
        int: The integer value of the parameter if found; otherwise, returns None.
    """

    val = None

    # Load the path to the gNodeB configuration file from config.yaml
    file_path = yaml.safe_load(open('config.yaml', 'r'))['bubbleran_network_setup'] + "/mx-conf/gnb.conf"

    # Read all lines from the gNodeB configuration file
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Search for the specified parameter and extract its integer value
    for i, line in enumerate(lines):
        if line.strip().startswith(var):
            val = int(line.split('=')[1].strip().strip(";"))
            break

    # Return the found value or None if not found
    return val



@tool
def calc_weighted_average(df: Annotated[dict, "Input DataFrame (in dict format)"], weight1: Annotated[float, "Weight for column index 1"],  weight2: Annotated[float, "Weight for column index 2"], current_param_value: Annotated[int, "Value to match in the first column (used as pivot)"]):
    """
    Calculates relative Weighted Average Gains based on a pivot parameter value.

    Args:
        df (dict): Input DataFrame (in dictionary format) with 3 columns:
                   - First column: Parameter values
                   - Second column: First KPI
                   - Third column: Second KPI
        weight1 (float): Weight applied to the second column (First KPI).
        weight2 (float): Weight applied to the third column (Second KPI).
        current_param_value (int): Parameter value to use as the pivot for percentage calculations.

    Returns:
        str: Formatted string of the resulting DataFrame showing percentage changes and weighted gains.
    """

    # Convert input dictionary to DataFrame
    df = pd.DataFrame(df)

    # Identify the pivot row based on the provided current_param_value
    pivot_row = df[df.iloc[:, 0] == current_param_value]
    new_data = []
    columns = df.columns.tolist()

    # Iterate through each row to calculate percentage changes relative to the pivot
    for _, row in df.iterrows():
        if row.iloc[0] !=current_param_value: 
            # Prevent division by zero by replacing zero denominators with 1
            denom_col1 = pivot_row[columns[1]].values[0] or 1
            denom_col2 = pivot_row[columns[2]].values[0] or 1

            # Calculate percentage increases for both columns
            changed_row = {
                f"Change_in_{columns[0]}_value": f"{current_param_value}_to_{row.iloc[0]}",
                f"%_increase_in_{columns[1]}": (row[columns[1]] - pivot_row[columns[1]].values[0])*100/denom_col1,
                f"%_increase_in_{columns[2]}": (row[columns[2]] - pivot_row[columns[2]].values[0])*100/denom_col2,
            }
            new_data.append(changed_row)
    # Create a DataFrame from the calculated percentage changes
    weighted_average_gain_df = pd.DataFrame(new_data)

    # Calculate the weighted average gain based on provided weights
    weighted_average_gain_df["Weighted_Average_Gain"] = (weight1* weighted_average_gain_df.iloc[:, 1]) + (weight2*weighted_average_gain_df.iloc[:,2]) 
    
    # Convert the DataFrame to a string without row indices for clean output
    weighted_average_gain_df = weighted_average_gain_df.to_string(index=False)
    return weighted_average_gain_df


@tool
def update_value_in_gnb(var: Annotated[str, "Parameter to update"], val: Annotated[int, "Integer value to set for the parameter"]):
    """
    Updates the specified configuration parameter in the gNodeB configuration file.

    Args:
        var (str): The name of the parameter to update.
        val (int): The new integer value to set for the parameter.

    """
    print(f"\n\n\nUPDATING {var} tp {val} in gnbtool")

    # Handle direct scalar parameter updates in gNodeB config (e.g., p0_nominal, att_tx, att_rx)
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
        target_files = [
            os.path.join(bubbleran_setup, "mx-conf", "gnb.conf"),
            os.path.join(bubbleran_setup, "mx-conf", "nr-rfsim.conf")
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

