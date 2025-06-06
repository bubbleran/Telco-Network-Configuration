"""
Network Performance Planner - Streamlit UI Controller

This is the main entry point for running the Telco LLM Workflow using Streamlit.

It provides an interactive UI for:
- Configuring network parameters.
- Starting and stopping the BubbleRAN network.
- Running agent-based monitoring, configuration, and validation loops.
- Reviewing KPI-based recommendations and applying changes.

Agents involved:
- Monitoring Agent: Observes network performance and detects degradation.
- Configuration Agent: Suggests optimal values for network parameters based on historical data.
- Validation Agent: Validates suggested changes in live network.

"""

import io
import csv
import time
import yaml
import sqlite3
import subprocess
import contextlib
from copy import deepcopy
from typing import Literal
from io import StringIO
import os

import pandas as pd
import streamlit as st

from langgraph.graph import StateGraph, START, END
from langchain_core.messages import convert_to_messages
from langchain_core.runnables import RunnableLambda

from agentic_llm_workflow.agents import State, config_agent, valid_agent, monitoring_agent, test_NIM
from agentic_llm_workflow.utils import update_value_in_gnb, update_value_in_db, start_network, add_traffic, stop_network, check_network_status, find_value_in_gnb, read_historical_data

# -------------------- Initialization --------------------

# Read and cache historical KPI data at startup
read_historical_data()    

# Initialize Streamlit session state variables if not already present
if "user_query" not in st.session_state:
    st.session_state.user_query = "" # maintains the user query to be sent to configuration agent

if "config_output" not in st.session_state:
    st.session_state.config_output = "" # maintains configuration output
    
if "show_validation_keys" not in st.session_state:
    st.session_state.show_validation_keys = False # flag to control visibility of validation buttons in the UI

# Function to find initial variable values, either from live running network, else fallback to config.yaml values
def find_init_var_val(var): 

    if check_network_status(): # if network is running 
        print(f"Network running. Reading {var} from gnb")
        return find_value_in_gnb(var)
    else:
        val = yaml.safe_load(open('config.yaml', 'r'))[f'default_{var}_value']
        update_value_in_gnb(var, val)
        print(f"Network not running. Reading default {var} from config")
        return val

if "global_state" not in st.session_state: # initialize global state with default or live network values
    st.session_state.global_state = [State()]
    st.session_state.global_state[0]["messages"] = []
    st.session_state.global_state[0]["vars_current"] = {
            "p0_nominal": find_init_var_val("p0_nominal"),
            "dl_carrierBandwidth": find_init_var_val("dl_carrierBandwidth"),
            "ul_carrierBandwidth": find_init_var_val("ul_carrierBandwidth"),
            "att_tx": find_init_var_val("att_tx"),
            "att_rx": find_init_var_val("att_rx"),
            }
    st.session_state.global_state[0]["vars_new"] = None

if "network_status" not in st.session_state:
    st.session_state.network_status = check_network_status()

# Determine if control buttons should be disabled based on network status. Controls the visibility of start and stop network buttons
disabled_button = not check_network_status()

# Additional session state initialization maintaining monitoring buttons, state and output
if "monitoring" not in st.session_state:
    st.session_state.monitoring = False

if "monitoring_output" not in st.session_state:
    st.session_state.monitoring_output = ""
    
def toggle_monitoring():
    st.session_state.monitoring = not st.session_state.monitoring

# -------------------- UI Setup --------------------

# Main page title and description
st.title("Network Performance Planner")
st.write("Telco agent-based LLM workflow that helps you reconfigure the network based on KPIs")
st.markdown("<br>", unsafe_allow_html=True)

# Build the agent workflow graph with Configuration and Validation agents
builder = StateGraph(State)
builder.add_edge(START, "Configuration Agent")
builder.add_node("Configuration Agent", config_agent)
builder.add_node("Validation Agent", valid_agent)
graph = builder.compile()

# If NIM_mode is True, check whether NIM is reachable and prompt user in case of any issues
if yaml.safe_load(open('config.yaml', 'r'))['NIM_mode']:
    response = test_NIM()
    if "Error" in response:
        st.error(response, icon="🚨")
        print("\n", response)
    

# -------------------- Sidebar Configuration Panel --------------------

## UI sidebar to configure values of parameters and monitoring/ validation wait times
with st.sidebar:
    st.title("Configuration")

    # Functions to synchronize DL/UL selections
    def change_dl_carrierBandwidth_val():
        st.session_state.selected_ul_dl_carrierBandwidth_value = st.session_state.selected_dl_carrierBandwidth_value

    def change_ul_carrierBandwidth_val():
        st.session_state.selected_ul_dl_carrierBandwidth_value = st.session_state.selected_ul_carrierBandwidth_value
    
    # Functions to update monitoring and validation wait times in config.yaml
    def update_monitoring_time():
        file_path = "config.yaml"
        with open(file_path, "r") as file:
            lines = file.readlines()
        with open(file_path, "w") as file:
            for line in lines:
                if "monitoring_wait_time:" in line: # Change value of variable in config.yaml
                    file.write(f"monitoring_wait_time: {st.session_state.monitoring_wait_value}\n")
                else:
                    file.write(line)  # Keep everything else unchanged
        print("Monitoring time updated successfully!")

    def update_validation_time():
        file_path = "config.yaml"
        with open(file_path, "r") as file:
            lines = file.readlines()
        with open(file_path, "w") as file:
            for line in lines:
                if "validation_wait_time:" in line: # Change value of variable in config.yaml
                    file.write(f"validation_wait_time: {st.session_state.validation_wait_value}\n")
                else:
                    file.write(line) # Keep everything else unchanged
        print("Validation time updated successfully!")

    # p0_nominal selection
    p0_nominal_values =  yaml.safe_load(open('config.yaml', 'r'))['p0_nominal_values']
    curr_p0_nominal = find_value_in_gnb("p0_nominal")
    selected_p0_nominal_value_box = st.selectbox("p0_nominal value", p0_nominal_values, key="selected_p0_nominal_value", index=p0_nominal_values.index(curr_p0_nominal))    

    # DL/UL carrier bandwidth selections
    dl_carrierBandwidth_values = yaml.safe_load(open('config.yaml', 'r'))['dl_carrierBandwidth_values']
    if "selected_ul_dl_carrierBandwidth_value" not in st.session_state:
        st.session_state.selected_ul_dl_carrierBandwidth_value = find_value_in_gnb("dl_carrierBandwidth")
    selected_dl_carrierBandwidth_value = st.selectbox("dl_carrierBandwidth value", dl_carrierBandwidth_values, index = dl_carrierBandwidth_values.index(st.session_state.selected_ul_dl_carrierBandwidth_value), key="selected_dl_carrierBandwidth_value", on_change=change_dl_carrierBandwidth_val)

    ul_carrierBandwidth_values = yaml.safe_load(open('config.yaml', 'r'))['ul_carrierBandwidth_values']
    selected_ul_carrierBandwidth_value_box = st.selectbox("ul_carrierBandwidth value", ul_carrierBandwidth_values, index = ul_carrierBandwidth_values.index(st.session_state.selected_ul_dl_carrierBandwidth_value), key="selected_ul_carrierBandwidth_value", on_change=change_ul_carrierBandwidth_val)

    # Additional parameter selections
    att_tx_values = yaml.safe_load(open('config.yaml', 'r'))['att_tx_values']
    curr_att_tx = find_value_in_gnb("att_tx")
    selected_att_tx_value_box = st.selectbox("att_tx value", att_tx_values, key="selected_att_tx_value", index=att_tx_values.index(curr_att_tx))#, on_change=change_att_tx_val)

    att_rx_values = yaml.safe_load(open('config.yaml', 'r'))['att_rx_values']
    curr_att_rx = find_value_in_gnb("att_rx")
    selected_att_rx_value_box = st.selectbox("att_rx value", att_rx_values, key="selected_att_rx_value", index=att_rx_values.index(curr_att_rx))#, on_change=change_att_rx_val)
    
    # Display warnings if configuration has changed but not reflected on the network (Restart of network pending)
    changed = False
    change_summary = []

    if selected_p0_nominal_value_box != st.session_state.global_state[-1]["vars_current"]["p0_nominal"]:
        changed = True
        change_summary.append("p0_nominal")

    if st.session_state.selected_ul_dl_carrierBandwidth_value != st.session_state.global_state[-1]["vars_current"]["dl_carrierBandwidth"]:
        changed = True
        change_summary.append("dl_carrierBandwidth")

    if st.session_state.selected_ul_dl_carrierBandwidth_value != st.session_state.global_state[-1]["vars_current"]["ul_carrierBandwidth"]:
        changed = True
        change_summary.append("ul_carrierBandwidth")

    if selected_att_tx_value_box != st.session_state.global_state[-1]["vars_current"]["att_tx"]:
        changed = True
        change_summary.append("att_tx")

    if selected_att_rx_value_box != st.session_state.global_state[-1]["vars_current"]["att_rx"]:
        changed = True
        change_summary.append("att_rx")

    if changed:
        changed_params = ", ".join(change_summary)
        st.warning(f"⚠️ {changed_params} updated. Stop network and restart to apply changes.")

    # Monitoring and validation wait time selections
    monitoring_wait_values = [1, 10, 30, 60]
    default_monitoring_wait_time = yaml.safe_load(open('config.yaml', 'r'))['monitoring_wait_time']
    monitoring_wait_value = st.selectbox("Monitoring time (s)", monitoring_wait_values, index=monitoring_wait_values.index(default_monitoring_wait_time), key="monitoring_wait_value", on_change=update_monitoring_time)

    validation_wait_values = [1, 10, 30, 60]
    default_validation_wait_time = yaml.safe_load(open('config.yaml', 'r'))['validation_wait_time']
    validation_wait_value = st.selectbox("Validation time (s)", validation_wait_values, index=validation_wait_values.index(default_validation_wait_time), key="validation_wait_value", on_change=update_validation_time)

    st.markdown("---")

    # Inputs for traffic simulation
    col1, col2 = st.columns(2)
    with col1:
        traffic_time = st.number_input("Traffic time (s)", min_value=0, value=15, step=1, disabled=disabled_button)
    
    with col2:
        traffic_bandwidth = st.number_input("Traffic bandwidth (M)", min_value=0, value=100, step=1, disabled=disabled_button)

    # Button to trigger traffic simulation
    if st.button("Add Traffic", use_container_width=True, disabled=disabled_button):
        if isinstance(traffic_time, (int, float)) and isinstance(traffic_bandwidth, (int, float)):
            traffic_result = add_traffic(traffic_time, traffic_bandwidth)
            if traffic_result.startswith("Success"):
                st.success("✅ Added traffic successfully")
            elif traffic_result.startswith("Error"):
                st.error("❌ Failed to add traffic")
            else:
                st.warning("⚠️ Please refer to README to add USRP traffic")
        else:
            st.error("Please enter valid value.")

    ## Control to reset all databases and network
    if st.button("Reset Application", key="reset_button", use_container_width=True, disabled=not disabled_button):
        try:
            if 'BUBBLERAN_HOST_PWD' not in os.environ:
                result = subprocess.run(["bash", "./agentic_llm_workflow/reset.sh"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            else:
                result = subprocess.run(["bash", "./agentic_llm_workflow/reset.sh"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            st.success("Reset complete")
        except subprocess.CalledProcessError as e:
            st.error(f"An error occurred during cleanup: {e}")
        time.sleep(1)
        st.rerun()


# -------------------- Network Control Buttons --------------------

left, right = st.columns([1,1])
# Button to start the network with updated configuration
if left.button("Start Network", use_container_width=True, disabled=not disabled_button):
    # Update global state with UI-selected values before starting
    st.session_state.global_state[-1]["vars_current"].update({
            "p0_nominal": st.session_state.selected_p0_nominal_value,
            "dl_carrierBandwidth": st.session_state.selected_dl_carrierBandwidth_value,
            "ul_carrierBandwidth": st.session_state.selected_ul_carrierBandwidth_value,
            "att_tx": st.session_state.selected_att_tx_value,
            "att_rx": st.session_state.selected_att_rx_value
        })

    # Update the values in gNB before starting network 
    update_value_in_gnb("p0_nominal", st.session_state.selected_p0_nominal_value)
    update_value_in_gnb("dl_carrierBandwidth", st.session_state.selected_dl_carrierBandwidth_value) # dl= ul for carrierBandwidth
    update_value_in_gnb("att_tx", st.session_state.selected_att_tx_value)
    update_value_in_gnb("att_rx", st.session_state.selected_att_rx_value)
    
    # Show command execution log in expandable section
    with st.expander("Running Commands..", expanded=True):
        responses = None
        for responses in start_network():
            st.write(responses)
            if responses.startswith("Success"):
                st.session_state.network_status = True
    # Provide feedback on network status
    if  st.session_state.network_status == False:
        st.error("❌ Failed to start network")
    else:
        
        st.success("✅ Network started successfully")
    time.sleep(2)
    st.rerun()


# Button to stop the network and save the latest KPIs in persistent database
if right.button("Stop Network", use_container_width=True, disabled=disabled_button):
    if st.session_state.monitoring:
        st.warning("Please stop monitoring and then stop the network.")
    else:
        try:
            update_value_in_db(st.session_state.global_state[-1]["vars_current"])
        except Exception as e:
            print(f"Warning: Could not sync persistent database with live database: {e}")
        response = ""
        with st.expander("Running Commands..", expanded=True):
            for response in stop_network():
                if response.startswith("Success"):
                    st.success(response)
                elif response.startswith("Error"):
                    st.error(response)
                else:
                    st.write(response)
        time.sleep(4)
        st.session_state.user_query = " "
        st.session_state.network_status = False
        st.rerun()

# -------------------- Monitoring Controls --------------------

# Dynamic label for starting/stopping monitoring
monitoring_button = "Stop monitoring" if st.session_state.monitoring else "Monitor network"
if st.button(monitoring_button, key="monitor", on_click=toggle_monitoring, use_container_width=True, disabled=disabled_button):
    pass  # Button toggles monitoring status

# Monitoring Agent loop when activated
if st.session_state.monitoring:
    st.session_state.user_query = " "
    st.session_state.config_output = ""

    expander_placeholder = st.expander("Monitoring Logs", expanded=True)  
    monitoring_output_placeholder = expander_placeholder.empty()
    all_outputs = []
    section_buffer = []

    while st.session_state.monitoring:
        monitoring_agent_response = None
        for monitoring_agent_response in monitoring_agent(st.session_state.global_state[-1]):
            if isinstance(monitoring_agent_response, dict):
                # last_monitoring_state = monitoring_agent_response
                st.session_state.global_state.append(monitoring_agent_response)
                all_outputs = []
                section_buffer = []
            else:
                if monitoring_agent_response == "reset":
                    all_outputs = []
                    section_buffer = []
                elif monitoring_agent_response.startswith("Error:"):
                    st.warning(monitoring_agent_response)
                    break
                else:
                    if "Monitoring" in monitoring_agent_response:
                        if section_buffer:
                            all_outputs.append("\n".join(section_buffer))
                            all_outputs.append("\n\n------\n\n")
                            section_buffer = []
                        # header = monitoring_agent_response.replace("Collecting KPIs for current", "### Monitoring").replace("for", "###\n\n• Collecting KPIs for")
                        section_buffer.append(monitoring_agent_response)
                    elif "Collecting" in monitoring_agent_response or "negative" in monitoring_agent_response:
                        section_buffer.append(f"{monitoring_agent_response}")
                    else:
                        section_buffer.pop()
                        section_buffer.append(f"{monitoring_agent_response}")

                    monitoring_output_placeholder.write("\n".join(all_outputs + section_buffer))
                    st.session_state.monitoring_output = monitoring_agent_response
            
        if section_buffer:
            all_outputs.append("\n".join(section_buffer))
            monitoring_output_placeholder.write("\n".join(all_outputs))
        
        # If the monitoring agent suggests to Call config agent:
        if "negative" in st.session_state.monitoring_output:
            # toggle_monitoring()
            var_current = st.session_state.monitoring_output.split()[-1]
            # Set the query to be sent to the Config Agent
            st.warning("Press Process Query button to continue with reconfiguration of parameter.")
            st.session_state.user_query = f"Suggest me the optimal value of {var_current}"
            st.session_state.monitoring = False
        time.sleep(1)
        
        
# -------------------- Configuration Controls --------------------
col1, col2 = st.columns([4, 1]) 
hide_enter_subtext = """
    <style>
    div[data-testid="InputInstructions"] > span:nth-child(1) {
        visibility: hidden;
    }
    </style>
    """
with col1:
    st.markdown(hide_enter_subtext, unsafe_allow_html=True)
    user_query = st.text_input(label="Process query", placeholder="Enter your query:", value=st.session_state.user_query, label_visibility='collapsed', disabled=disabled_button)

with col2:
    st.markdown(hide_enter_subtext, unsafe_allow_html=True)
    process_button = st.button("Process Query", key = "process", disabled=disabled_button) 


# Define textbox to show the outputs from config agent
config_output_placeholder = st.empty()

# Function to update transcript from agent Uudates
def update_config_transcript(update, status=None):
    
    if isinstance(update, tuple):
        ns, update = update
        if len(ns) == 0:
            return
    for node_name, node_update in update.items():
        for m in convert_to_messages(node_update["messages"]):
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                m.pretty_print()
            st.session_state.config_output += buffer.getvalue() + "\n"
            with config_output_placeholder.expander("Configuration Agent Transcript", expanded=True):
                st.text_area(label="  ", value=st.session_state.config_output, height=200,)   

# Function to generate summary of the Configuration Agent transcript
def summarize_config_output(output_dict):
    try:
        average_kpis_df = pd.read_csv(StringIO(output_dict["average_kpis_df"]), sep=r'\s+')
        average_kpis_df.reset_index(drop=True, inplace=True)

        weighted_average_gain = pd.read_csv(StringIO(output_dict["weighted_average_gain"]), sep=r'\s+')
        weighted_average_gain.reset_index(drop=True, inplace=True)

        var = average_kpis_df.columns[0]
        optimal_value = output_dict["vars_new"][var]

        kpi1 = average_kpis_df.iloc[:, 1]
        kpi1 = kpi1.name.replace('Average_of_', '')

        kpi2 = average_kpis_df.iloc[:, 2]
        kpi2 = kpi2.name.replace('Average_of_', '')

        average_kpis_df.columns = average_kpis_df.columns.str.replace('_', ' ')
        weighted_average_gain.columns = weighted_average_gain.columns.str.replace('_', ' ')

        first_col = weighted_average_gain.columns[0]
        weighted_average_gain[first_col] = weighted_average_gain[first_col].str.replace('_to_', ' to ')
        weighted_average_gain[first_col] = weighted_average_gain[first_col].str.replace('.0', '', regex=False)

        map_kpi_headings = {"bitrate_DL": "DL Bitrate", "bitrate_UL": "UL Bitrate", "snr": "Layer 1 SNR", "retx": "Number of Retransmissions requested by the UE"}
        map_kpis = {"bitrate_DL": "DL Bitrate", "bitrate_UL": "UL Bitrate", "snr": "SNR", "retx": "Number of Retransmissions requested by the UE"}

        if optimal_value!=output_dict["vars_current"][var]:
            append_text = "You can now choose to either validate with these values, or choose your own preference on the via the Configuration sidebar in UI directly."
        else:
            append_text = "Since you already have the optimal value, no other steps are needed."
        
        container_style = """
            border: none;
            border-radius: 0;
            padding: 5px 0;
            margin: 5px 0;
            background-color: transparent;
            box-shadow: none;
            font-size: 14px;
        """
        with st.expander(f"Summary Report of {var} Optimization", expanded=False):
            with st.container():
                st.markdown(f"""
                <div style="{container_style}">
                    <p>To suggest an optimal <strong>{var}</strong> value, I rely on two KPIs: <strong>{map_kpi_headings.get(kpi1)}</strong> (primary goal) and <strong>{map_kpi_headings.get(kpi2)}</strong> (secondary trade-off KPI).</p>
                    <p><strong>Here's the approach I use:</strong></p>
                    <ul>
                        <li><strong>KPI-Based Evaluation:</strong>  
                            I evaluate the effect of different <strong>{var}</strong> values on both <strong>{map_kpis.get(kpi1)}</strong> and <strong>{map_kpis.get(kpi2)}</strong>.  
                            The goal is to maximize <strong>{map_kpis.get(kpi1)}</strong> while maintaining acceptable <strong>{map_kpis.get(kpi2)}</strong> levels.
                        </li>
                        <li><strong>Data Analysis:</strong>  
                            I have gathered historical data of <strong>{map_kpis.get(kpi1)}</strong> and <strong>{map_kpis.get(kpi2)}</strong> for each <strong>{var}</strong> setting.  
                            A table below illustrates the average value of the KPIs collected for {st.session_state.monitoring_wait_value}s for different value of {var}. 
                        </li>
                    </ul>
                </div>""", unsafe_allow_html=True)
        
                col1, col2 = st.columns([0.90, 0.10])
                with col1:
                    st.markdown(f"**KPI Comparison Table:**")
                with col2:
                    with st.popover("ⓘ", use_container_width=True):
                        st.markdown(f"<div style='font-size: 14px;'>You can select the desired {var} value directly via the UI,<br>based on your preferred KPI trade-offs shown in this table.</div>",
                        unsafe_allow_html=True)

                st.dataframe(average_kpis_df.reset_index(drop=True), use_container_width=True)
                st.markdown(f"""
                <div style="{container_style}">
                    <ul>
                        <li><strong>Weighted KPI Trade-off Calculation:</strong>  
                            I use a weighted average to combine the average change in <strong>{map_kpis.get(kpi1)}</strong> and the corresponding change in <strong>{map_kpis.get(kpi2)}</strong>.  
                            I am using the values given in the <code>config.yaml</code> file to assign weights {yaml.safe_load(open('config.yaml', 'r'))[f'{var}_WA_weights'][0]} and {yaml.safe_load(open('config.yaml', 'r'))[f'{var}_WA_weights'][1]} to the two KPIs respectively.
                        </li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
                col1, col2 = st.columns([0.90, 0.10])
                with col1:
                    st.markdown("**Weighted Average Table:**")
                with col2:
                    with st.popover("ⓘ", use_container_width=True):
                        st.markdown(
                            "<div style='font-size: 14px;'>In my future release, you can modify the weights <br>according to your priorities interactively.</div>",
                            unsafe_allow_html=True
                            )

                
                st.dataframe(weighted_average_gain.reset_index(drop=True), use_container_width=True)
                st.markdown(f"""
                <div style="{container_style}">
                    <p><strong>Optimal Suggestion:</strong>  
                    Based on the data, the value that provides the optimal weighted average is <strong>{optimal_value}</strong>.</p>
                    <p>{append_text}</p>
                </div>
                """, unsafe_allow_html=True)

    except Exception as e:
        print(f"Error: Unable to generate summary: {e}")

# When the user submits the query (i.e., process_button is clicked), stream the user query to graph config agent
if process_button:
    # Clear previous config output and validation keys
    if st.session_state.show_validation_keys: 
        st.session_state.show_validation_keys = False
    st.session_state.config_output = ""

    if user_query:
        # Prepare new state for the graph execution with user query
        st.session_state.show_validation_keys = False
        s = None
        new_state = deepcopy(st.session_state.global_state[-1])
        new_state["vars_new"] = None
        new_state["messages"].append(("user", user_query))
        new_state["agent_id"] = "human"
        st.session_state.global_state.append(new_state)
        
        # Stream updates from the graph and display them in the transcript
        for s in graph.stream(new_state, subgraphs=True):
            update_config_transcript(s)
        
        last_state = s[1]['Configuration Agent']
        st.session_state.global_state.append(last_state)

        # Show final configuration agent output
        with config_output_placeholder.expander("Configuration Agent Transcript", expanded=False):
            st.text_area(
                label=" ",
                value=st.session_state.config_output,
                height=200,
            )
        
        # Summarize or display error based on agent result
        if last_state["vars_new"]:
            st.success("Summary generated. Click below to expand report.")
            summarize_config_output(last_state)
        else:
            st.error(last_state["messages"][-1][1].strip("Error:").strip())
            time.sleep(2)
        
        # If config agent has successfully generated recommended parameters, show keys for validation agent
        if last_state["vars_new"] and last_state["vars_new"]!=last_state["vars_current"]:
            st.session_state.show_validation_keys = True

    # user_query = ""
        
# -------------------- Validation Controls --------------------

if st.session_state.show_validation_keys:
    config_output_placeholder = st.empty()
    st.session_state.config_output = ""

    # Button to apply changes and validate
    left, right = st.columns([1,1])
    if left.button("Make Changes and Validate", use_container_width=True):
        expander_placeholder = st.expander("Validation Logs", expanded=True)
        validation_output_placeholder = expander_placeholder.empty()
        all_outputs = []
        for responses in valid_agent(st.session_state.global_state[-1]):
            if isinstance(responses, dict):
                st.session_state.global_state.append(responses)
            else:
                all_outputs.append(str(responses))
            validation_output_placeholder.write("\n\n".join(all_outputs))

        # Update UI state and reset validation flag
        last_state = st.session_state.global_state[-1]
        st.session_state.selected_ul_dl_carrierBandwidth_value = find_value_in_gnb("dl_carrierBandwidth")#last_state["vars_current"]["dl_carrierBandwidth"]
        time.sleep(7)
        st.session_state.show_validation_keys = False
        st.session_state.global_state[-1]["vars_new"] = None
        st.session_state.global_state[-1]["vars_current"] = {
            "p0_nominal": find_init_var_val("p0_nominal"),
            "dl_carrierBandwidth": find_init_var_val("dl_carrierBandwidth"),
            "ul_carrierBandwidth": find_init_var_val("ul_carrierBandwidth"),
            "att_tx": find_init_var_val("att_tx"),
            "att_rx": find_init_var_val("att_rx"),
            }
        st.session_state.config_output = ""
        st.session_state.user_query = " "
        st.rerun()
        
    # Button to ignore suggested changes
    if right.button("Ignore suggestions", use_container_width=True):
        st.session_state.show_validation_keys = False
        st.session_state.global_state[-1]["vars_new"] = None 
        st.session_state.config_output = ""
        st.success("Reverted back to old configuration")
        time.sleep(7)
        st.session_state.user_query = " "
        st.rerun()



