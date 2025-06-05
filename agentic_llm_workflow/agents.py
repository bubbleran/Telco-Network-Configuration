"""
LangGraph Agent Orchestration for BubbleRAN

This file defines the behavior of three key agents used in the blueprint pipeline:
- Configuration Agent
- Validation Agent
- Monitoring Agent

These agents work together to:
- Interpret user queries and determine relevant network parameters.
- Apply proposed parameter changes to the gNodeB configuration.
- Collect, aggregate, and analyze network KPIs through SQL queries.
- Calculate weighted average gains to validate configuration decisions.
- Revert to stable configurations if performance degrades.

The agents ensure data-driven and adaptive network optimization in a controlled, explainable manner.
"""

import json
import sqlite3
import time
from copy import deepcopy
from io import StringIO
import os
import requests

import pandas as pd
import streamlit as st
import yaml
from pydantic import BaseModel
from typing import Annotated, Dict, Literal, Optional, Union
from typing_extensions import TypedDict

import langgraph
from langgraph.graph import END, MessagesState, StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent
from langchain_core.prompts import PromptTemplate
from langgraph.types import Command

from langchain_core.messages import AnyMessage, HumanMessage, convert_to_messages
from langchain_core.tools import tool
from langchain_nvidia_ai_endpoints import ChatNVIDIA

from agentic_llm_workflow.tools import ( calc_weighted_average, execute_historical_sql, execute_xapp_sql, find_value_in_gnb)

from agentic_llm_workflow.utils import (check_network_status, start_network, stop_network, update_value_in_db, update_value_in_gnb)

class State(TypedDict):
    """
    Defines the structure for maintaining state in the LangGraph workflow.
    """
    next: str
    agent_id: str
    messages: Annotated[list, add_messages] = []
    average_kpis_df : Optional[pd.DataFrame]
    weighted_average_gain : Optional[pd.DataFrame]
    vars_current : Dict[str, int] # Maintains current values of all parameters
    vars_new : Dict[str, int] # Maintains new potential values of all parameters


def init_agent():
    """
    Initializes the ChatNVIDIA agent based on runtime configuration.

    Returns:
        ChatNVIDIA: An instance configured for either NIM or direct API usage.
    """
    config = None
    with open('config.yaml', 'r') as config_file:
        config = yaml.safe_load(config_file)

    # Determine if NIM mode is inactive based on configuration
    if not yaml.safe_load(open('config.yaml', 'r'))['NIM_mode']:
        # Initialize the LLM for direct API usage
        llm = ChatNVIDIA(
            model=yaml.safe_load(open('config.yaml', 'r'))['llm_model'],
            api_key=yaml.safe_load(open('config.yaml', 'r'))['nvidia_api_key'], 
            base_url=yaml.safe_load(open('config.yaml', 'r'))['llm_base_url'],
            temperature=yaml.safe_load(open('config.yaml', 'r'))['llm_temp'],
            top_p=yaml.safe_load(open('config.yaml', 'r'))['llm_top_p'],
            max_tokens=yaml.safe_load(open('config.yaml', 'r'))['llm_max_tokens'],
            )
    else: 
        # Extract model and runtime details for NIM
        nim_image = yaml.safe_load(open('config.yaml', 'r'))['nim_image']
        nim_llm_model = nim_image.split("nvcr.io/nim/")[-1].split(":")[0]
        nim_llm_port = yaml.safe_load(open('config.yaml', 'r'))['nim_llm_port']
        if "BUBBLERAN_HOST_PWD" in os.environ:
            nim_base_url = f"http://host.docker.internal:{nim_llm_port}/v1"
        else:
            nim_base_url = f"http://localhost:{nim_llm_port}/v1"

        # Initialize the LLM for NIM 
        llm = ChatNVIDIA(
            model=nim_llm_model,
            api_key=yaml.safe_load(open('config.yaml', 'r'))['nvidia_api_key'], 
            base_url=nim_base_url,
            temperature=yaml.safe_load(open('config.yaml', 'r'))['llm_temp'],
            top_p=yaml.safe_load(open('config.yaml', 'r'))['llm_top_p'],
            max_tokens=yaml.safe_load(open('config.yaml', 'r'))['llm_max_tokens'],
            )
    return llm


def monitoring_agent(state: State) -> State:
    """
    5G Network Monitoring Agent for LangGraph.
    ------------------------------------------
    This agent sequentially monitors the following 5G parameters:
    - p0_nominal
    - dl_carrierBandwidth
    - ul_carrierBandwidth
    - att_tx
    - att_rx

    For each parameter, it:
    - Collects KPIs for a fixed monitoring duration (as set up in UI or config.yaml d).
    - Aggregates performance data through SQL queries.
    - Calculates weighted average gain.
    - Decides whether to escalate to a config agent.
    """

    print("\n\nInside Monitoring Agent")

    # Initialize LLM agent with system prompt and tools
    llm = init_agent()
    system_prompt = 'You are an agent in a LangGraph. Your task is to help a user configure or analyze a current 5G network. \
        Reply concisely and exactly in the format directed to you.'
    llm_agent =  create_react_agent(llm, tools=[execute_xapp_sql, calc_weighted_average], prompt=system_prompt)
    
    # Load static configurations from config.yaml
    config = yaml.safe_load(open('config.yaml', 'r'))
    monitoring_wait_time = config['monitoring_wait_time']

    ######################################################################################################################################################
    ## Monitor Weighted Average Gain for p0_nominal:

    # try:
    if True:
        param = "p0_nominal"
        p0_nominal_current = state["vars_current"][param]
        yield f"Monitoring {param} = {p0_nominal_current}: "
        weight1, weight2 = config[f'{param}_WA_weights']

        yield f"Collecting KPIs for {monitoring_wait_time} seconds..."
        time.sleep(monitoring_wait_time)

        update_value_in_db(state["vars_current"])
        yield "Aggregating data based on current value..."

        # Build SQL query prompt
        prompt = f''' Your task is to create a SQL query, execute it using the execute_xapp_sql tool EXACTLY ONCE, and return the resulting answer dataframe. \
            Do not make multiple calls to the tool. 
            There is a table called "{config['table_name']}" with columns "tstamp", "pusch_snr", "p0_nominal", "dl_aggr_tbs".
            Write the SQL query which does the following:
            1. Makes another new column called "bitrate_dl", such that bitrate_dl[i] = max(0, (1000* (dl_aggr_tbs[i]-dl_aggr_tbs[i-1]))/(tstamp[i]-tstamp[i-1]))
            2. Makes a new column called "snr", such that snr[i] = pusch_snr[i]
            3. Finds the average of "bitrate_dl" and "snr" ,for these values of "p0_nominal" : {p0_nominal_current}
            The SQL query should return me these columns in this order: "p0_nominal","Average_of_bitrate_dl", "Average_of_snr". 
            You should only print the dataframe, with column names, received from the execute_xapp_sql tool in following format: "data_frame : ,explanation: "'''

        # Invoke LLM agent
        llm_response = llm_agent.invoke({"messages": prompt})
        average_kpis_df = llm_response["messages"][-2].content if hasattr(llm_response["messages"][-1], "content") else None
        # print(average_kpis_df)

        if not average_kpis_df:
            raise ValueError("Error: LLM agent did not return a valid dataframe output.")

        # Parse results
        average_kpis_df = pd.read_csv(StringIO(average_kpis_df), sep=r'\s+')
        average_kpis_df["weighted_avg_gain"] = average_kpis_df[average_kpis_df.columns.tolist()[1]]*(weight1)+average_kpis_df[average_kpis_df.columns.tolist()[2]]*(weight2) 
        weighted_avg_gain = average_kpis_df["weighted_avg_gain"].values[0]

        yield f"Weighted Average Gain observed: {weighted_avg_gain:.2f}"
        
        if float(weighted_avg_gain)<0:
        # if True:
            message = f"\n⚠️ Weighted average gain is negative. Press Process Query button to reconfigure {param}."
            yield message
            print("Exiting Monitoring Agent\n\n")
            yield {"next": "config_agent", "agent_id": "monitoring_agent", "messages": [('assistant',message)], "average_kpis_df": average_kpis_df, "vars_current": state["vars_current"], "vars_new": None, "weighted_average_gain": weighted_avg_gain}
            return

    # except Exception as e:
    #     error_message = f"Error: Monitoring Agent encountered an error on {param} due to {str(e)}\n"
    #     print(error_message)
    #     yield error_message
    #     return
    
    ######################################################################################################################################################
    ## Monitor Weighted Average Gain for dl_carrierBandwidth::

    try:
        param = "dl_carrierBandwidth"
        dl_carrierBandwidth_current = state["vars_current"][param] 
        yield f"Monitoring {param} = {dl_carrierBandwidth_current}: "
        weight1, weight2 = config[f'{param}_WA_weights']
        
        yield f"Collecting KPIs for {monitoring_wait_time} seconds..."
        time.sleep(monitoring_wait_time)

        update_value_in_db(state["vars_current"])
        yield "Aggregating data based on current value..."

        # Build SQL query prompt
        prompt = f''' Your task is to create a SQL query, execute it using the execute_xapp_sql tool EXACTLY ONCE, and return the resulting answer dataframe. \
            Do not make multiple calls to the tool. 
            There is a table called "{config['table_name']}" with columns "tstamp", "pusch_snr", "dl_carrierBandwidth", "dl_aggr_tbs".
            Write the SQL query which does the following:
            1. Makes another new column called "bitrate_dl", such that bitrate_dl[i] = max(0, (1000* (dl_aggr_tbs[i]-dl_aggr_tbs[i-1]))/(tstamp[i]-tstamp[i-1]))
            2. Makes a new column called "snr", such that snr[i] = pusch_snr[i]
            3. Finds the average of "bitrate_dl" and "snr" ,for these values of "dl_carrierBandwidth" : {dl_carrierBandwidth_current}
            The SQL query should return me these columns in this order: "dl_carrierBandwidth","Average_of_bitrate_dl", "Average_of_snr". 
            You should only print the dataframe, with column names, received from the execute_xapp_sql tool in following format: "data_frame : ,explanation: "'''

        # Invoke LLM agent
        llm_response = llm_agent.invoke({"messages": prompt})
        average_kpis_df = llm_response["messages"][-2].content if hasattr(llm_response["messages"][-1], "content") else None
        # print(average_kpis_df)
        
        if not average_kpis_df:
            raise ValueError("Error: LLM agent did not return a valid dataframe output.")

        # Parse results
        average_kpis_df = pd.read_csv(StringIO(average_kpis_df), sep=r'\s+')
        average_kpis_df["weighted_avg_gain"] = average_kpis_df[average_kpis_df.columns.tolist()[1]]*(weight1)+average_kpis_df[average_kpis_df.columns.tolist()[2]]*(weight2) 
        weighted_avg_gain = average_kpis_df["weighted_avg_gain"].values[0]

        yield f"Weighted Average Gain observed: {weighted_avg_gain:.2f}"

        # Check if reconfiguration is needed
        if float(weighted_avg_gain)<0:
            message = f"\n⚠️ Weighted average gain is negative. Press Process Query button to reconfigure {param}."
            yield message
            print("Exiting Monitoring Agent\n\n")
            yield {"next": "config_agent", "agent_id": "monitoring_agent", "messages": [('assistant',message)], "average_kpis_df": None, "vars_current": state["vars_current"], "vars_new": None, "weighted_average_gain": None}
            return

    except Exception as e:
        error_message = f"Error: Monitoring Agent encountered an error on {param} due to {str(e)}\n"
        print(error_message)
        yield error_message
        return

    ######################################################################################################################################################
    ## Monitor Weighted Average Gain for ul_carrierBandwidth:

    try:
        param = "ul_carrierBandwidth"
        ul_carrierBandwidth_current = state["vars_current"][param]
        yield f"Monitoring {param} = {ul_carrierBandwidth_current}: "
        weight1, weight2 = config[f'{param}_WA_weights']

        yield f"Collecting KPIs for {monitoring_wait_time} seconds..."
        time.sleep(monitoring_wait_time)

        update_value_in_db(state["vars_current"])
        yield "Aggregating data based on current value..."

        # Build SQL query prompt
        prompt = f''' Your task is to create a SQL query, execute it using the execute_xapp_sql tool EXACTLY ONCE, and return the resulting answer dataframe. \
            Do not make multiple calls to the tool. 
            There is a table called "{config['table_name']}" with columns "tstamp", "pusch_snr", "ul_carrierBandwidth", "ul_aggr_tbs".
            Write the SQL query which does the following:
            1. Makes another new column called "bitrate_ul", such that bitrate_ul[i] = max(0, (1000* (ul_aggr_tbs[i]-ul_aggr_tbs[i-1]))/(tstamp[i]-tstamp[i-1]))
            2. Makes a new column called "snr", such that snr[i] = pusch_snr[i]
            3. Finds the average of "bitrate_ul" and "snr" ,for these values of "ul_carrierBandwidth" : {ul_carrierBandwidth_current} 
            The SQL query should return me these columns in this order: "ul_carrierBandwidth","Average_of_bitrate_ul", "Average_of_snr". 
            You should only print the dataframe, with column names, received from the execute_xapp_sql tool in following format: "data_frame : ,explanation: "'''

        # Invoke LLM agent
        llm_response = llm_agent.invoke({"messages": prompt})
        average_kpis_df = llm_response["messages"][-2].content if hasattr(llm_response["messages"][-1], "content") else None
        # print(average_kpis_df)
        
        if not average_kpis_df:
                raise ValueError("Error: LLM agent did not return a valid dataframe output.")

        # Parse results
        average_kpis_df = pd.read_csv(StringIO(average_kpis_df), sep=r'\s+')
        average_kpis_df["weighted_avg_gain"] = average_kpis_df[average_kpis_df.columns.tolist()[1]]*(weight1)+average_kpis_df[average_kpis_df.columns.tolist()[2]]*(weight2) 
        weighted_avg_gain = average_kpis_df["weighted_avg_gain"].values[0]

        yield f"Weighted Average Gain observed: {weighted_avg_gain:.2f}"

        # Check if reconfiguration is needed
        if float(weighted_avg_gain)<0:
            message = f"\n⚠️ Weighted average gain is negative. Press Process Query button to reconfigure {param}."
            yield message
            print("Exiting Monitoring Agent\n\n")
            yield {"next": "config_agent", "agent_id": "monitoring_agent", "messages": [('assistant',message)], "average_kpis_df": None, "vars_current": state["vars_current"], "vars_new": None, "weighted_average_gain": None}
            return

    except Exception as e:
        error_message = f"Error: Monitoring Agent encountered an error on {param} due to {str(e)}\n"
        print(error_message)
        yield error_message
        return

    ######################################################################################################################################################
    ## Monitor Weighted Average Gain for att_tx:

    try:
        param = "att_tx"
        att_tx_current = state["vars_current"][param]
        yield f"Monitoring {param} = {att_tx_current}: "
        weight1, weight2 = config[f'{param}_WA_weights']

        yield f"Collecting KPIs for {monitoring_wait_time} seconds..."
        time.sleep(monitoring_wait_time)

        update_value_in_db(state["vars_current"])
        yield "Aggregating data based on current value..."

        # Build SQL query prompt
        prompt = f''' Your task is to create a SQL query, execute it using the execute_xapp_sql tool EXACTLY ONCE, and return the resulting answer dataframe. \
            Do not make multiple calls to the tool. 
            There is a table called "{config['table_name']}" with columns "tstamp", "pusch_snr", "att_tx", "dl_aggr_tbs".
            Write the SQL query which does the following:
            1. Makes another new column called "bitrate_dl", such that bitrate_dl[i] = max(0, (1000* (dl_aggr_tbs[i]-dl_aggr_tbs[i-1]))/(tstamp[i]-tstamp[i-1]))
            2. Makes a new column called "snr", such that snr[i] = pusch_snr[i]
            3. Finds the average of "bitrate_dl" and "snr" ,for these values of "att_tx" : {att_tx_current}
            The SQL query should return me these columns in this order: "att_tx","Average_of_bitrate_dl", "Average_of_snr". 
            You should only print the dataframe, with column names, received from the execute_xapp_sql tool in following format: "data_frame : ,explanation: "'''

        # Invoke LLM agent
        llm_response = llm_agent.invoke({"messages": prompt})
        average_kpis_df = llm_response["messages"][-2].content if hasattr(llm_response["messages"][-1], "content") else None
        # print(average_kpis_df)
        
        if not average_kpis_df:
                raise ValueError("Error: LLM agent did not return a valid dataframe output.")

        # Parse results
        average_kpis_df = pd.read_csv(StringIO(average_kpis_df), sep=r'\s+')
        average_kpis_df["weighted_avg_gain"] = average_kpis_df[average_kpis_df.columns.tolist()[1]]*(weight1)+average_kpis_df[average_kpis_df.columns.tolist()[2]]*(weight2) 
        weighted_avg_gain = average_kpis_df["weighted_avg_gain"].values[0]

        yield f"Weighted Average Gain observed: {weighted_avg_gain:.2f}"

        # Check if reconfiguration is needed
        if float(weighted_avg_gain)<0:
            message = f"\n⚠️ Weighted average gain is negative. Press Process Query button to reconfigure {param}."
            yield message
            print("Exiting Monitoring Agent\n\n")
            yield {"next": "config_agent", "agent_id": "monitoring_agent", "messages": [('assistant',message)], "average_kpis_df": None, "vars_current": state["vars_current"], "vars_new": None, "weighted_average_gain": None}
            return

    except Exception as e:
        error_message = f"Error: Monitoring Agent encountered an error on {param} due to {str(e)}\n"
        print(error_message)
        yield error_message
        return

    ######################################################################################################################################################
    ## Monitor Weighted Average Gain for att_rx:

    try:
        param = "att_rx"
        att_rx_current = state["vars_current"][param]
        yield f"Monitoring {param} = {att_rx_current}: "
        weight1, weight2 = config[f'{param}_WA_weights']

        yield f"Collecting KPIs for {monitoring_wait_time} seconds..."
        time.sleep(monitoring_wait_time)
        
        update_value_in_db(state["vars_current"])
        yield "Aggregating data based on current value..."

        # Build SQL query prompt
        prompt = f''' Your task is to create a SQL query, execute it using the execute_xapp_sql tool EXACTLY ONCE, and return the resulting answer dataframe. \
            Do not make multiple calls to the tool. 
            There is a table called "{config['table_name']}" with columns "tstamp", "dl_harq_round1", "dl_harq_round2", "dl_harq_round3", "att_rx", "ul_aggr_tbs".
            Write the SQL query which does the following:
            1. Makes another new column called "bitrate_ul", such that bitrate_ul[i] = max(0, (1000* (ul_aggr_tbs[i]-ul_aggr_tbs[i-1]))/(tstamp[i]-tstamp[i-1]))
            2. Makes a new column called "retx", such that retx[i] = max(0, (1000*(dl_harq_round0[i] + dl_harq_round1[i] + dl_harq_round2[i] + dl_harq_round3[i] - dl_harq_round0[i-1] - dl_harq_round1[i-1] - dl_harq_round2[i-1] - dl_harq_round3[i-1] ))/ (tstamp[i]-tstamp[i-1]))
            3. Finds the average of "bitrate_ul" and "retx" ,for these values of "att_rx" : {att_rx_current}
            The SQL query should return me these columns in this order: "att_rx","Average_of_bitrate_ul", "Average_of_retx". 
            You should only print the dataframe, with column names, received from the execute_xapp_sql tool in following format: "data_frame : ,explanation: "'''

        # Invoke LLM agent            
        llm_response = llm_agent.invoke({"messages": prompt})
        average_kpis_df = llm_response["messages"][-2].content if hasattr(llm_response["messages"][-1], "content") else None
        # print(average_kpis_df)

        if not average_kpis_df:
                raise ValueError("Error: LLM agent did not return a valid dataframe output.")
        
        # Parse results
        average_kpis_df = pd.read_csv(StringIO(average_kpis_df), sep=r'\s+')
        average_kpis_df["weighted_avg_gain"] = average_kpis_df[average_kpis_df.columns.tolist()[1]]*(weight1)+average_kpis_df[average_kpis_df.columns.tolist()[2]]*(weight2) 
        weighted_avg_gain = average_kpis_df["weighted_avg_gain"].values[0]

        yield f"Weighted Average Gain observed: {weighted_avg_gain:.2f}"

        # Check if reconfiguration is needed
        if float(weighted_avg_gain)<0:
            message = f"\n⚠️ Weighted average gain is negative. Press Process Query button to reconfigure {param}."
            yield message
            print("Exiting Monitoring Agent\n\n")
            yield {"next": "config_agent", "agent_id": "monitoring_agent", "messages": [('assistant',message)], "average_kpis_df": None, "vars_current": state["vars_current"], "vars_new": None, "weighted_average_gain": None}
            return

    except Exception as e:
        error_message = f"Error: Monitoring Agent encountered an error on {param} due to {str(e)}\n"
        print(error_message)
        yield error_message
        return

    ########################################
    # Send command to clear monitoring output
    yield "reset" 
    return


def config_agent(state: State) -> State:
    """
    5G Network Configuration Agent for LangGraph.
    ---------------------------------------------
    This agent analyzes and suggests configuration adjustments for the supported 5G parameters.

    For each parameter, it:
    - Interprets user queries to detect the target parameter.
    - Collects and aggregrates historical KPIs using SQL queries.
    - Calculates weighted average gain based on user-defined weights.
    - Recommends whether a parameter change is beneficial.
    """
    
    # Initialize LLM Agent based on the config.yaml parameters
    llm = init_agent()

    # Create LLM agent with specified tools and behavior modifier
    system_prompt = 'You are an agent in a LangGraph. Your task is to help an user configure/ analyse a current 5G network. \
        You must reply to the questions asked concisely, and exactly in the format directed to you.'
    llm_agent =  create_react_agent(llm, tools=[find_value_in_gnb, execute_historical_sql, calc_weighted_average], prompt=system_prompt)
    
    # Extract the last message, which is the user query
    user_query = state["messages"][-1]
    vars_new = deepcopy(state["vars_current"])
    config = yaml.safe_load(open('config.yaml', 'r'))

    # Compose the initial prompt to classify the user query into a network parameter category
    # prompt0 = f'''You are given a list of TOPICS and a USER QUERY below. 
    # Your job is to respond to the user query based on given RULES. Your answer should consist of Reason, followed by Topic.
    
    # TOPIC: [p0_nominal, dl_carrierBandwidth, ul_carrierBandwidth, att_tx, att_rx, Others].
    
    # USER QUERY: {user_query}

    # Answer the user query based on these RULES strictly:
    # 1. If the user asks what is the current value of a given parameter from the list, call the `find_value_in_gnb` to answer what is the value of the parameter. Make sure to say Topic: Others.
    # 2. If the user wants to find the optimal value of a given parameter, say for example, p0_nominal, ONLY then you should answer "Topic: p0_nominal", or "Topic: att_tx", and so on.
    # 3. If the user asks anything outside topic, answer politely and tell you can only handle queries about the network and return "Other" as the query topic.
    
    # Output structure should be strictly as follows:
    # Reason:
    # Topic: detected_TOPIC
    
    # Reason step by step before answering.'''
    prompt0 = f'''
    You will receive a USER QUERY and a list of PARAMETERS. 
    Your task is to classify the query into the correct topic, following the \
    RULES below. Your response must include a clear step-by-step Reason, \
    followed by the Parameter.

    PARAMETERS: p0_nominal, dl_carrierBandwidth, ul_carrierBandwidth, att_tx, att_rx, Others
    
    USER QUERY: {user_query}

    RULES:
    1. If the user asks for the current value of a parameter from the list:
    Use the tool find_value_in_gnb(parameter_name) to retrieve the value.
    Provide the retrieved value in your response.
    Set Parameter: Others.

    2. If the user asks for the optimal value or best setting of a parameter from the list:
    Return the corresponding Parameter, such as "Parameter: p0_nominal", "Parameter: att_tx", \
    "Parameter: att_rx", "Parameter: dl_carrierBandwidth" or "Parameter: ul_carrierBandwidth".

    3. If the query is unrelated to these network topics:
    Respond politely that you can only handle network parameter queries.
    Set Parameter: Others.

    OUTPUT FORMAT:
    Reason:
    <Step-by-step reasoning here>

    Parameter: <detected_TOPIC>
    '''

    llm_response = llm_agent.invoke({"messages": prompt0})
    param = llm_response["messages"][-1].content

    try:
        # Compose the initial prompt to classify the user query into a network parameter category
        param = param.split()[-1].strip().strip(".,{}()'\"")
    except Exception as e:
        param = "Others"
    
    if param=="p0_nominal":

        try:
            # Prompt the agent to extract the current value of p0_nominal from user query
            prompt1 = f'''You are given a user query. Understand what is the current value of {param} selected by the user. 
                You should ONLY output the value of {param} exclusively, for example: '103', or '-10. 
                USER_QUERY: ${user_query}. 
                If you do not find the {param} value in this query, you can call the tool called `find_value_in_gnb` to find the value of the parameter.
                To call the `find
                Make sure that your final output should just be the integer value of the respective parameter.
                
                Sample outputs:
                value: -106
                value: 90
                value: 12

                Make sure to follow the sample output format very strictly.'''

            llm_response = llm_agent.invoke({"messages": prompt1})
            p0_nominal = int(llm_response["messages"][-1].content.split(":")[-1].strip().strip(".").strip("(){}"))
            if p0_nominal not in config["p0_nominal_values"]:
                raise ValueError(f"Extracted value {param} is not in the allowed values {config['p0_nominal_values']}")


        except Exception as e:
            error_message = f"""Error: 🚨 Failed to extract current {param} value from LLM response.
                LLM Raw Response: {llm_response["messages"][-1].content}
                Error: {str(e)}"""
            print(error_message)
            return {"next": None, "agent_id": "config_agent", "messages": state["messages"]+[("assistant", error_message)], "average_kpis_df":None, "weighted_average_gain":None, "vars_current": state["vars_current"], "vars_new": None}
            raise ValueError(error_message.strip())

        try:
            # Prompt the agent to construct and execute SQL query on historical database
            prompt2 = f''' Your task is to create a SQL query, execute it using the execute_historical_sql tool to get the proper result dataframe, and return the resulting answer dataframe. \
            If you get error, make another call with proper SQL to the tool to get proper answer.
            There is a table called "kpis" with columns "Parameter", "Value", "snr", "bitrate_DL". \
            Write an SQL query which does the following:
                1. Filter rows where "Parameter" value is "P0 Nominal".
                2. Find average of "bitrate_DL" and "snr", FOR EACH of these distinct "Value" separately: {config['p0_nominal_values']}.  
            The SQL query should return me these columns in the given order: "p0_nominal", "Average_of_bitrate_DL", "Average_of_snr". 
            Also add filter to remove the rows where "snr" and "bitrate_DL" values are 0. \
            You should only print the dataframe, with column names, received from the execute_historical_sql tool in following format: "data_frame : ,explanation: "
            If your first call gets error, make sure to rerun with full SQL again to get proper response from the SQL Query.'''

            llm_response = llm_agent.invoke({"messages": prompt2})
            # print(llm_response)
            average_kpis_df = llm_response["messages"][-2].content if hasattr(llm_response["messages"][-1], "content") else None
            print("results dataframe: ", average_kpis_df)

        except Exception as e:
            error_message = f"""Error: 🚨 Failed to create or run SQL on historical database.
                LLM Raw Response: {llm_response["messages"][-1].content}
                Error: {str(e)}"""
            print(error_message)
            return {"next": None, "agent_id": "config_agent", "messages": state["messages"]+[("assistant", error_message)], "average_kpis_df":None, "weighted_average_gain":None, "vars_current": state["vars_current"], "vars_new": None}
            raise ValueError(error_message.strip())

        try:
            # Prompt the agent to compute weighted average gain using SQL results
            weight1, weight2 = config[f'{param}_WA_weights']
            prompt3 = f''' Your task is to calculate the weighted average gain using the `calc_weighted_average` tool. The tool accepts the following parameters:
            
            Here are the inputs:
            1. `data_frame`: 
            {average_kpis_df}
            2. `weight1`: {weight1}
            3. `weight2`: {weight2}
            4. `current_param_value`: {p0_nominal}
            Use the tool to calculate the weighted average gain. Return **only** the resulting DataFrame as the output.'''

            llm_response = llm_agent.invoke({"messages":[prompt3]})
            weighted_avg = llm_response['messages'][-2].content

        except Exception as e:
            error_message = f"""Error: 🚨 Failed to calculate average weighted gain table. Please make sure to provide correct value of {param}.
                LLM Raw Response: {llm_response["messages"][-1].content}
                Error: {str(e)}"""
            print(error_message)
            return {"next": None, "agent_id": "config_agent", "messages": state["messages"]+[("assistant", error_message)], "average_kpis_df": average_kpis_df, "weighted_average_gain":None, "vars_current": state["vars_current"], "vars_new": None}
            raise ValueError(error_message.strip())
            
        try:
            # Generate the final recommendation based on the weighted average table
            prompt4 = f'''The table contains the following columns: [Change in P0_nominal, % increase in bitrate_DL, % increase snr, Weighted Average Gain]. Use only the data in the table to answer questions. Do not guess or provide information beyond the table.

            "Change in P0_nominal" column shows the potential change of P0_nominal value. Example: "-106 to -102" means if they value is changed from -106 to -102, the wieghted average gain is given in the corresponding row.
            
            You are given that the current P0_Nominal value is {p0_nominal}.

            Here is the table which shows expected gain/ loss upon changing the P0_nominal value from the {p0_nominal} to other values:
            {weighted_avg}

            Based on the given table, answer the user's question. USER QUESTION: "{user_query}"

            Instructions for answering:
            1. The final call on which P0_Nominal value or which change in P0_Nominal value is preferable is determined by whose 'Weighted Average Gain' value is the **greatest**.
            2. Refer to the table explicitly for your answer. Think step by step before answering. First think the reason, and then give the answer.
            3. The '% increase in SNR' increase is favoured, and is given weight {weight2} to calculate the weighted average gain.
            4. The '% increase Bit Rate' is favoured and given a weight of {weight1} in the weighted average gain.
            5. Negative Weighted Average Gain suggest that the P0_nominal change is **not** recommended. If all changes (like -106 to -102, and so on) result in negative Weighted Average Gains, inform the user that no change is needed.

            Think step by step. Answer in this format: "Reason:, Answer: new_P0_value"
            The "Answer" should be the new p0 nominal value ONLY. If there is no change needed, write the current/ the old p0 nominal value only.
            **DO NOT use any tools**

            Here is a sample response:
            "There are currently 4 different values configured in the data available to me for P0_nominal. \
            Considering the KPIs resulting from these values, the optimal value is X. \
            There are two main KPIs used in the decision making process. The bitrate and SNR. \
            The tradeoff between SNR increase and \
            bitrate increase is calculated with Weighted average method and the best performing value compared to the current P0 nominal value, is -86."
            '''

            llm_response = llm_agent.invoke({"messages": prompt4}) 
            vars_new["p0_nominal"] = int(float(llm_response["messages"][-1].content.split('Answer: ')[1].split(',')[0]))
            # print(llm_response[""])
            # Return success state with updated parameter suggestion
            return {"next": "Validation Agent", "agent_id": "config_agent", "messages": state["messages"]+[("assistant", llm_response["messages"][-1].content)], "average_kpis_df": average_kpis_df, "weighted_average_gain":weighted_avg, "vars_current": state["vars_current"], "vars_new": vars_new}

        except Exception as e:
            error_message = f"""Error: 🚨 Failed to extract new value of {param}. 
                LLM Raw Response: {llm_response["messages"][-1].content}
                Error: {str(e)}"""
            print(error_message)
            return {"next": None, "agent_id": "config_agent", "messages": state["messages"]+[("assistant", error_message)], "average_kpis_df": average_kpis_df, "weighted_average_gain":weighted_avg, "vars_current": state["vars_current"], "vars_new": None}


    elif param=="dl_carrierBandwidth":
        try:
            # Prompt the agent to extract the current value of dl_carrierBandwidth from user query
            prompt1 = f'''You are given a user query. Understand what is the current value of {param} selected by the user. 
                You should ONLY output the value of {param} exclusively, for example: '103', or '-10. 
                USER_QUERY: ${user_query}. 
                If you do not find the {param} value in this query, you can call the tool called `find_value_in_gnb` to find the value of the parameter.
                Make sure that your final output should just be the integer value of the respective parameter.
                
                Sample outputs:
                value: -106
                value: 90
                value: 12
                
                Make sure to follow the sample output format strictly.'''
            llm_response = llm_agent.invoke({"messages": prompt1})
            dl_carrierBandwidth = int(llm_response["messages"][-1].content.split(":")[-1].strip().strip(".").strip("(){}"))
        
        except Exception as e:
            error_message = f"""Error: 🚨 Failed to extract current {param} value from LLM response.
                LLM Raw Response: {llm_response["messages"][-1].content}
                Error: {str(e)}"""
            print(error_message)
            return {"next": None, "agent_id": "config_agent", "messages": state["messages"]+[("assistant", error_message)], "average_kpis_df":None, "weighted_average_gain":None, "vars_current": state["vars_current"], "vars_new": None}
            raise ValueError(error_message.strip())
        
        try:
            # Prompt the agent to construct and execute SQL query on historical database
            prompt2 = f''' Your task is to create a SQL query, execute it using the execute_historical_sql tool to get the proper result dataframe, and return the resulting answer dataframe. \
            If you get error, make another call with proper SQL to the tool to get proper answer.
            There is a table called "kpis" with columns "Parameter", "Value", "snr", "bitrate_DL". \
            Write an SQL query which does the following:
                1. Filter rows where "Parameter" value is "Number of Physical Resource Blocks (PRBs)".
                2. Find average of "bitrate_DL" and "snr", FOR EACH of these distinct "Value" separately: {config['dl_carrierBandwidth_values']}. 
            The SQL query should return me these columns in the given order: "dl_carrierBandwidth", "Average_of_bitrate_DL", "Average_of_snr". 
            Also add filter to remove the rows where "snr" and "bitrate_DL" values are 0. \
            You should only print the dataframe, with column names, received from the execute_historical_sql tool in following format: "data_frame : ,explanation: "
            If your first call gets error, make sure to rerun with full SQL again to get proper response from the SQL Query.'''

            llm_response = llm_agent.invoke({"messages": prompt2})
            # print(llm_response)
            average_kpis_df = llm_response["messages"][-2].content if hasattr(llm_response["messages"][-1], "content") else None
            print("results dataframe: ", average_kpis_df)

        except Exception as e:
            error_message = f"""Error: 🚨 Failed to create or run SQL on historical database.
                LLM Raw Response: {llm_response["messages"][-1].content}
                Error: {str(e)}"""
            print(error_message)
            return {"next": None, "agent_id": "config_agent", "messages": state["messages"]+[("assistant", error_message)], "average_kpis_df":None, "weighted_average_gain":None, "vars_current": state["vars_current"], "vars_new": None}
            raise ValueError(error_message.strip())

        try:
            # Prompt the agent to compute weighted average gain using SQL results
            weight1, weight2 = config[f'{param}_WA_weights']
            prompt3 = f''' Your task is to calculate the weighted average gain using the `calc_weighted_average` tool. The tool accepts the following parameters:
            Here are the inputs:
            1. `data_frame`: 
            {average_kpis_df}
            2. `weight1`: {weight1}
            3. `weight2`: {weight2}
            4. `current_param_value`: {dl_carrierBandwidth}
            Use the tool to calculate the weighted average gain. Return **only** the resulting DataFrame as the output.'''
            
            llm_response = llm_agent.invoke({"messages":[prompt3]})
            weighted_avg = llm_response['messages'][-2].content

        except Exception as e:
            error_message = f"""Error: 🚨 Failed to calculate average weighted gain table. Please make sure to provide correct value of {param}.
                LLM Raw Response: {llm_response["messages"][-1].content}
                Error: {str(e)}"""
            print(error_message)
            return {"next": None, "agent_id": "config_agent", "messages": state["messages"]+[("assistant", error_message)], "average_kpis_df": average_kpis_df, "weighted_average_gain":None, "vars_current": state["vars_current"], "vars_new": None}
            raise ValueError(error_message.strip())
        
        try:
            # Generate the final recommendation based on the weighted average table
            prompt4 = f'''The table contains the following columns: [Change in dl_carrierBandwidth, % increase in snr (sound noise ratio), % increase Bit Rate, Weighted Average Gain]. Use only the data in the table to answer questions. Do not guess or provide information beyond the table.

            "Change in dl_carrierBandwidth" column shows the potential change of dl_carrierBandwidth value. Example: "-106 to -102" means if they value is changed from -106 to -102, the wieghted average gain is given in the corresponding row.
            
            You are given that the current dl_carrierBandwidth value is {dl_carrierBandwidth}.

            Here is the table which shows expected gain/ loss upon changing the dl_carrierBandwidth value from the {dl_carrierBandwidth} to other values:
            {weighted_avg}

            Based on the given table, answer the user's question. USER QUESTION: "{user_query}"

            Instructions for answering:
            1. The final call on which dl_carrierBandwidth value or which change in dl_carrierBandwidth value is preferable is determined by whose 'Weighted Average Gain' value is the **greatest**.
            2. Refer to the table explicitly for your answer. Think step by step before answering. First think the reason, and then give the answer.
            3. The '% increase in Sound Noise Ratio (SNR)' increase causes UL interference, and is given weight {weight2} to calculate the weighted average gain.
            4. The '% increase Bit Rate' is favoured and given a weight of {weight1} in the weighted average gain.
            5. Negative Weighted Average Gain suggest that the dl_carrierBandwidth change is **not** recommended. If all changes (like -106 to -102, and so on) result in negative Weighted Average Gains, inform the user that no change is needed.

            Think step by step. Answer in this format: "Reason:, Answer: new_dl_carrierBandwidth"
            The "Answer" should be the new_dl_carrierBandwidth value ONLY. If there is no change needed, write the current/ the old p0 nominal value only.
            **DO NOT use any tools**

            Here is a sample response:
            "There are currently 5 different values configured in the data available to me for dl_carrierBandwidth. \
            Considering the KPIs resulting from these values, the optimal value is X. \
            There are two main KPIs used in the decision making process. The bitrate and SNR. \
            As the SNR increase causes UL interference, the tradeoff between SNR increase and \
            bitrate increase is calculated with Weighted average method and the best performing value compared to the current P0 nominal value, is -84."
            '''

            llm_response = llm_agent.invoke({"messages": prompt4})
            vars_new["dl_carrierBandwidth"]  = int(float(llm_response["messages"][-1].content.split('Answer: ')[1].split(',')[0]))
            return {"next": "Validation Agent", "agent_id": "config_agent", "messages": state["messages"]+[("assistant", llm_response["messages"][-1].content)], "average_kpis_df": average_kpis_df, "weighted_average_gain":weighted_avg, "vars_current": state["vars_current"], "vars_new": vars_new}

        except Exception as e:
            error_message = f"""Error: 🚨 Failed to extract new value of {param}. 
                LLM Raw Response: {llm_response["messages"][-1].content}
                Error: {str(e)}"""
            print(error_message)
            return {"next": None, "agent_id": "config_agent", "messages": state["messages"]+[("assistant", error_message)], "average_kpis_df": average_kpis_df, "weighted_average_gain":weighted_avg, "vars_current": state["vars_current"], "vars_new": None}

    elif param=="ul_carrierBandwidth":
        try:
            # Prompt the agent to extract the current value of ul_carrierBandwidth from user query
            prompt1 = f'''You are given a user query. Understand what is the current value of {param} selected by the user. 
                You should ONLY output the value of {param} exclusively, for example: '103', or '-10. 
                USER_QUERY: ${user_query}. 
                If you do not find the {param} value in this query, you can call the tool called `find_value_in_gnb` to find the value of the parameter.
                Make sure that your final output should just be the integer value of the respective parameter.
                
                Sample outputs:
                value: -106
                value: 90
                value: 12
                
                Make sure to follow the sample output format strictly.'''
            
            llm_response = llm_agent.invoke({"messages": prompt1})
            ul_carrierBandwidth = int(llm_response["messages"][-1].content.split(":")[-1].strip().strip(".").strip("(){}"))

        except Exception as e:
            error_message = f"""Error: 🚨 Failed to extract current {param} value from LLM response.
                LLM Raw Response: {llm_response["messages"][-1].content}
                Error: {str(e)}"""
            print(error_message)
            return {"next": None, "agent_id": "config_agent", "messages": state["messages"]+[("assistant", error_message)], "average_kpis_df":None, "weighted_average_gain":None, "vars_current": state["vars_current"], "vars_new": None}
            raise ValueError(error_message.strip())

        try:
            # Prompt the agent to construct and execute SQL query on historical database
            prompt2 = f''' Your task is to create a SQL query, execute it using the execute_historical_sql tool to get the proper result dataframe, and return the resulting answer dataframe. \
            If you get error, make another call with proper SQL to the tool to get proper answer.
            There is a table called "kpis" with columns "Parameter", "Value", "snr", "bitrate_UL". \
            Write an SQL query which does the following:
                1. Filter rows where "Parameter" value is "UL_Number of Physical Resource Blocks (PRBs)".
                2. Find average of "bitrate_UL" and "snr", FOR EACH of these distinct "Value" separately: {config['ul_carrierBandwidth_values']}
            The SQL query should return me these columns in the given order: "ul_carrierBandwidth", "Average_of_bitrate_UL", "Average_of_snr". 
            Also add filter to remove the rows where "snr" and "bitrate_UL" values are 0. \
            You should only print the dataframe, with column names, received from the execute_historical_sql tool in following format: "data_frame : ,explanation: "
            If your first call gets error, make sure to rerun with full SQL again to get proper response from the SQL Query.'''

            llm_response = llm_agent.invoke({"messages": prompt2})
            # print(llm_response)
            average_kpis_df = llm_response["messages"][-2].content if hasattr(llm_response["messages"][-1], "content") else None
            print("results dataframe: ", average_kpis_df)

        except Exception as e:
            error_message = f"""Error: 🚨 Failed to create or run SQL on historical database.
                LLM Raw Response: {llm_response["messages"][-1].content}
                Error: {str(e)}"""
            print(error_message)
            return {"next": None, "agent_id": "config_agent", "messages": state["messages"]+[("assistant", error_message)], "average_kpis_df":None, "weighted_average_gain":None, "vars_current": state["vars_current"], "vars_new": None}
            raise ValueError(error_message.strip())
        
        try:
            # Prompt the agent to compute weighted average gain using SQL results
            weight1, weight2 = config[f'{param}_WA_weights']
            prompt3 = f''' Your task is to calculate the weighted average gain using the `calc_weighted_average` tool. The tool accepts the following parameters:
            Here are the inputs:
            1. `data_frame`: 
            {average_kpis_df}
            2. `weight1`: {weight1}
            3. `weight2`: {weight2}
            4. `current_param_value`: {ul_carrierBandwidth}
            Use the tool to calculate the weighted average gain. Return **only** the resulting DataFrame as the output.'''
            
            llm_response = llm_agent.invoke({"messages":[prompt3]})
            weighted_avg = llm_response['messages'][-2].content

        except Exception as e:
            error_message = f"""Error: 🚨 Failed to calculate average weighted gain table. Please make sure to provide correct value of {param}.
                LLM Raw Response: {llm_response["messages"][-1].content}
                Error: {str(e)}"""
            print(error_message)
            return {"next": None, "agent_id": "config_agent", "messages": state["messages"]+[("assistant", error_message)], "average_kpis_df": average_kpis_df, "weighted_average_gain":None, "vars_current": state["vars_current"], "vars_new": None}
            raise ValueError(error_message.strip())
        
        try:
            # Generate the final recommendation based on the weighted average table
            prompt4 = f'''The table contains the following columns: [Change in ul_carrierBandwidth, % increase in snr (sound noise ratio), % increase Bit Rate, Weighted Average Gain]. Use only the data in the table to answer questions. Do not guess or provide information beyond the table.

            "Change in ul_carrierBandwidth" column shows the potential change of ul_carrierBandwidth value. Example: "-106 to -102" means if they value is changed from -106 to -102, the wieghted average gain is given in the corresponding row.
            
            You are given that the current ul_carrierBandwidth value is {ul_carrierBandwidth}.

            Here is the table which shows expected gain/ loss upon changing the ul_carrierBandwidth value from the {ul_carrierBandwidth} to other values:
            {weighted_avg}

            Based on the given table, answer the user's question. USER QUESTION: "{user_query}"

            Instructions for answering:
            1. The final call on which ul_carrierBandwidth value or which change in ul_carrierBandwidth value is preferable is determined by whose 'Weighted Average Gain' value is the **greatest**.
            2. Refer to the table explicitly for your answer. Think step by step before answering. First think the reason, and then give the answer.
            3. The '% increase in Sound Noise Ratio (SNR)' increase causes UL interference, and is given weight {weight2} to calculate the weighted average gain.
            4. The '% increase Bit Rate' is favoured and given a weight of {weight1} in the weighted average gain.
            5. Negative Weighted Average Gain suggest that the ul_carrierBandwidth change is **not** recommended. If all changes (like -106 to -102, and so on) result in negative Weighted Average Gains, inform the user that no change is needed.

            Think step by step. Answer in this format: "Reason:, Answer: new_ul_carrierBandwidth"
            The "Answer" should be the new_ul_carrierBandwidth value ONLY. If there is no change needed, write the current/ the old p0 nominal value only.
            **DO NOT use any tools**

            Here is a sample response:
            "There are currently 5 different values configured in the data available to me for ul_carrierBandwidth. \
            Considering the KPIs resulting from these values, the optimal value is X. \
            There are two main KPIs used in the decision making process. The bitrate and SNR. \
            As the SNR increase causes UL interference, the tradeoff between SNR increase and \
            itrate increase is calculated with Weighted average method and the best performing value compared to the current P0 nominal value, is -84."
            '''

            llm_response = llm_agent.invoke({"messages": prompt4})
            vars_new["ul_carrierBandwidth"] = int(float(llm_response["messages"][-1].content.split('Answer: ')[1].split(',')[0]))
            return {"next": "Validation Agent", "agent_id": "config_agent", "messages": state["messages"]+[("assistant", llm_response["messages"][-1].content)], "average_kpis_df": average_kpis_df, "weighted_average_gain":weighted_avg, "vars_current": state["vars_current"], "vars_new": vars_new}

        except Exception as e:
            error_message = f"""Error: 🚨 Failed to extract new value of {param}. 
                LLM Raw Response: {llm_response["messages"][-1].content}
                Error: {str(e)}"""
            print(error_message)
            return {"next": None, "agent_id": "config_agent", "messages": state["messages"]+[("assistant", error_message)], "average_kpis_df": average_kpis_df, "weighted_average_gain":weighted_avg, "vars_current": state["vars_current"], "vars_new": None}


    elif param=="att_tx":
        
        try:        
            # Prompt the agent to extract the current value of att_tx from user query
            weight1, weight2 = config[f'{param}_WA_weights']
            prompt1 = f'''You are given a user query. Understand what is the current value of {param} selected by the user. 
                You should ONLY output the value of {param} exclusively, for example: '103', or '-10. 
                USER_QUERY: ${user_query}. 
                If you do not find the {param} value in this query, you can call the tool called `find_value_in_gnb` to find the value of the parameter.
                Make sure that your final output should just be the integer value of the respective parameter.
                
                Sample outputs:
                value: -106
                value: 90
                value: 12
                
                Make sure to follow the sample output format strictly.'''

            llm_response = llm_agent.invoke({"messages": prompt1})
            att_tx = int(llm_response["messages"][-1].content.split(":")[-1].strip().strip(".").strip("(){}"))

        except Exception as e:
            error_message = f"""Error: 🚨 Failed to extract current {param} value from LLM response.
                LLM Raw Response: {llm_response["messages"][-1].content}
                Error: {str(e)}"""
            print(error_message)
            return {"next": None, "agent_id": "config_agent", "messages": state["messages"]+[("assistant", error_message)], "average_kpis_df":None, "weighted_average_gain":None, "vars_current": state["vars_current"], "vars_new": None}
            raise ValueError(error_message.strip())

        try:
            # Prompt the agent to construct and execute SQL query on historical database
            prompt2 = f''' Your task is to create a SQL query, execute it using the execute_historical_sql tool to get the proper result dataframe, and return the resulting answer dataframe. \
            If you get error, make another call with proper SQL to the tool to get proper answer.
            There is a table called "kpis" with columns "Parameter", "Value", "snr", "bitrate_DL". \
            The Value column shows the value of the corresponding Parameter.
            Write an SQL query which does the following:
                1. Filter rows where "Parameter" value is "att_tx".
                2. Find average of "bitrate_DL" and "snr", FOR EACH of these distinct "Value" separately: {config['att_tx_values']}. 
            The SQL query should return me these columns in the given order: "att_tx", "Average_of_bitrate_DL", "Average_of_snr". 
            Order rows by DESCENDING order of "att_tx"
            Also add filter to remove the rows where "snr" and "bitrate_DL" values are 0. 
            You should only print the dataframe, with column names, received from the execute_historical_sql tool in following format: "data_frame : ,explanation: "
            If your first call gets error, make sure to rerun with full SQL again to get proper response from the SQL Query.'''

            llm_response = llm_agent.invoke({"messages": prompt2})
            average_kpis_df = llm_response["messages"][-2].content if hasattr(llm_response["messages"][-1], "content") else None
        except Exception as e:
            error_message = f"""Error: 🚨 Failed to create or run SQL on historical database.
                LLM Raw Response: {llm_response["messages"][-1].content}
                Error: {str(e)}"""
            print(error_message)
            return {"next": None, "agent_id": "config_agent", "messages": state["messages"]+[("assistant", error_message)], "average_kpis_df":None, "weighted_average_gain":None, "vars_current": state["vars_current"], "vars_new": None}
            raise ValueError(error_message.strip())

        try:
            # Prompt the agent to compute weighted average gain using SQL results
            prompt3 = f''' Your task is to calculate the weighted average gain using the `calc_weighted_average` tool. The tool accepts the following parameters:
            Think properly before calling tool, and make sure to call with right dataframe. Here are the inputs:
            1. `data_frame`: 
            
            {average_kpis_df}
            
            The dataframe consists of three columns named att_tx_value, Average_of_bitrate_DL and Average_of_snr, send all the three columns properly. 
            2. `weight1`: {weight1}
            3. `weight2`: {weight2}
            4. `current_param_value`: {att_tx}
            Use the tool to calculate the weighted average gain. Return **only** the resulting DataFrame as the output.'''
        
            llm_response = llm_agent.invoke({"messages":[prompt3]})
            # print(llm_response)
            weighted_avg = llm_response['messages'][-2].content

        except Exception as e:
            error_message = f"""Error: 🚨 Failed to calculate average weighted gain table. Please make sure to provide correct value of {param}.
                LLM Raw Response: {llm_response["messages"][-1].content}
                Error: {str(e)}"""
            print(error_message)
            return {"next": None, "agent_id": "config_agent", "messages": state["messages"]+[("assistant", error_message)], "average_kpis_df": average_kpis_df, "weighted_average_gain":None, "vars_current": state["vars_current"], "vars_new": None}
            raise ValueError(error_message.strip())
        
        try:
            # Generate the final recommendation based on the weighted average table
            prompt4 = f'''The table contains the following columns: [Change in att_tx, % increase in snr (sound noise ratio), % increase Bit Rate, Weighted Average Gain]. Use only the data in the table to answer questions. Do not guess or provide information beyond the table.

            "Change in att_tx" column shows the potential change of att_tx value. Example: "-106 to -102" means if they value is changed from -106 to -102, the wieghted average gain is given in the corresponding row.
            
            You are given that the current att_tx value is {att_tx}.

            Here is the table which shows expected gain/ loss upon changing the att_tx value from the {att_tx} to other values:
            {weighted_avg}

            Based on the given table, answer the user's question. USER QUESTION: "{user_query}"

            Instructions for answering:
            1. The final call on which att_tx value or which change in att_tx value is preferable is determined by whose 'Weighted Average Gain' value is the **greatest**.
            2. Refer to the table explicitly for your answer. Think step by step before answering. First think the reason, and then give the answer.
            3. The '% increase in Sound Noise Ratio (SNR)' increase causes UL interference, and is given weight {weight2} to calculate the weighted average gain.
            4. The '% increase Bit Rate' is favoured and given a weight of {weight1} in the weighted average gain.
            5. Negative Weighted Average Gain suggest that the att_tx change is **not** recommended. If all changes (like -106 to -102, and so on) result in negative Weighted Average Gains, inform the user that no change is needed.

            Think step by step. Answer in this format: "Reason:, Answer: new_att_tx"
            The "Answer" should be the new_att_tx value ONLY. If there is no change needed, write the current/ the old p0 nominal value only.
            **DO NOT use any tools**

            Here is a sample response:
            "There are currently 5 different values configured in the data available to me for att_tx. \
            Considering the KPIs resulting from these values, the optimal value is X. \
            There are two main KPIs used in the decision making process. The bitrate and SNR. \
            As the SNR increase causes UL interference, the tradeoff between SNR increase and \
            bitrate increase is calculated with Weighted average method and the best performing value compared to the current P0 nominal value, is -84."
            '''

            llm_response = llm_agent.invoke({"messages": prompt4})
            vars_new["att_tx"] = int(float(llm_response["messages"][-1].content.split('Answer: ')[1].split(',')[0]))
            return {"next": "Validation Agent", "agent_id": "config_agent", "messages": state["messages"]+[("assistant", llm_response["messages"][-1].content)], "average_kpis_df": average_kpis_df, "weighted_average_gain":weighted_avg, "vars_current": state["vars_current"], "vars_new": vars_new}

        except Exception as e:
            error_message = f"""Error: 🚨 Failed to extract new value of {param}. 
                LLM Raw Response: {llm_response["messages"][-1].content}
                Error: {str(e)}"""
            print(error_message)
            return {"next": None, "agent_id": "config_agent", "messages": state["messages"]+[("assistant", error_message)], "average_kpis_df": average_kpis_df, "weighted_average_gain":weighted_avg, "vars_current": state["vars_current"], "vars_new": None}

    elif param=="att_rx":

        try:
            # Prompt the agent to extract the current value of att_tx from user query
            prompt1 = f'''You are given a user query. Understand what is the current value of {param} selected by the user. 
            You should ONLY output the value of {param} exclusively, for example: '103', or '-10. 
            USER_QUERY: ${user_query}. 
            If you do not find the {param} value in this query, you can call the tool called `find_value_in_gnb` to find the value of the parameter.
            Make sure that your final output should just be the integer value of the respective parameter.
            
            Sample outputs:
            value: -106
            value: 90
            value: 12
            
            Make sure to follow the sample output format strictly.'''

            llm_response = llm_agent.invoke({"messages": prompt1})
            att_rx = int(llm_response["messages"][-1].content.split(":")[-1].strip().strip(".").strip("(){}"))

        except Exception as e:
            error_message = f"""Error: 🚨 Failed to extract current {param} value from LLM response.
                LLM Raw Response: {llm_response["messages"][-1].content}
                Error: {str(e)}"""
            print(error_message)
            return {"next": None, "agent_id": "config_agent", "messages": state["messages"]+[("assistant", error_message)], "average_kpis_df":None, "weighted_average_gain":None, "vars_current": state["vars_current"], "vars_new": None}
            raise ValueError(error_message.strip())

        try:
            # Prompt the agent to construct and execute SQL query on historical database
            prompt2 = f''' Your task is to create a SQL query, execute it using the execute_historical_sql tool to get the proper result dataframe, and return the resulting answer dataframe. \
            If you get error, make another call with proper SQL to the tool to get proper answer.
            There is a table called "kpis" with columns "Parameter", "Value", "retx", "bitrate_UL". 
            Write an SQL query which does the following:
                1. Filter rows where "Parameter" value is "att_rx".
                2. Find average of "bitrate_UL" and "retx", FOR EACH of these distinct "Value" separately: {config['att_rx_values']}
            The SQL query should return me these columns in the given order: "att_rx", "Average_of_bitrate_UL", "Average_of_retx". 
            Order rows by DESCENDING order of "att_rx"
            Also add filter to remove the rows where "retx" and "bitrate_UL" values are 0. 
            You should only print the dataframe, with column names, received from the execute_historical_sql tool in following format: "data_frame : ,explanation: "
            If your first call gets error, make sure to rerun with full SQL again to get proper response from the SQL Query.'''

            llm_response = llm_agent.invoke({"messages": prompt2})
            # print(llm_response)
            average_kpis_df = llm_response["messages"][-2].content if hasattr(llm_response["messages"][-1], "content") else None
            print("results dataframe: ", average_kpis_df)
        except Exception as e:
            error_message = f"""Error: 🚨 Failed to create or run SQL on historical database.
                LLM Raw Response: {llm_response["messages"][-1].content}
                Error: {str(e)}"""
            print(error_message)
            return {"next": None, "agent_id": "config_agent", "messages": state["messages"]+[("assistant", error_message)], "average_kpis_df":None, "weighted_average_gain":None, "vars_current": state["vars_current"], "vars_new": None}
            raise ValueError(error_message.strip())

        try:
            # Prompt the agent to compute weighted average gain using SQL results
            weight1, weight2 = config[f'{param}_WA_weights']
            prompt3 = f''' Your task is to calculate the weighted average gain using the `calc_weighted_average` tool. The tool accepts the following parameters:
            Here are the inputs:
            1. `data_frame`: 
            
            {average_kpis_df} 
            
            ## Input dataframe in dict format consisting of att_rx_value, Average_of_bitrate_UL and Average_of_retx fields
            2. `weight1`: {weight1}
            3. `weight2`: {weight2}
            4. `current_param_value`: {att_rx}
            Use the tool to calculate the weighted average gain. Return **only** the resulting DataFrame as the output.'''
            # print(prompt3)
            llm_response = llm_agent.invoke({"messages":[prompt3]})
            weighted_avg = llm_response['messages'][-2].content
        
        except Exception as e:
            error_message = f"""Error: 🚨 Failed to calculate average weighted gain table. Please make sure to provide correct value of {param}.
                LLM Raw Response: {llm_response["messages"][-1].content}
                Error: {str(e)}"""
            print(error_message)
            return {"next": None, "agent_id": "config_agent", "messages": state["messages"]+[("assistant", error_message)], "average_kpis_df": average_kpis_df, "weighted_average_gain":None, "vars_current": state["vars_current"], "vars_new": None}
            raise ValueError(error_message.strip())
        
        try:        
            # Generate the final recommendation based on the weighted average table
            prompt4 = f'''The table contains the following columns: [Change in att_rx, % increase in snr (sound noise ratio), % increase Bit Rate, Weighted Average Gain]. Use only the data in the table to answer questions. Do not guess or provide information beyond the table.

            "Change in att_rx" column shows the potential change of att_rx value. Example: "-106 to -102" means if they value is changed from -106 to -102, the wieghted average gain is given in the corresponding row.
            
            You are given that the current att_rx value is {att_rx}.

            Here is the table which shows expected gain/ loss upon changing the att_rx value from the {att_rx} to other values:
            {weighted_avg}

            Based on the given table, answer the user's question. USER QUESTION: "{user_query}"

            Instructions for answering:
            1. The final call on which att_rx value or which change in att_rx value is preferable is determined by whose 'Weighted Average Gain' value is the **greatest**.
            2. Refer to the table explicitly for your answer. Think step by step before answering. First think the reason, and then give the answer.
            3. The '% increase in Sound Noise Ratio (SNR)' increase causes UL interference, and is given weight {weight2} to calculate the weighted average gain.
            4. The '% increase Bit Rate' is favoured and given a weight of {weight1} in the weighted average gain.
            5. Negative Weighted Average Gain suggest that the att_rx change is **not** recommended. If all changes (like -106 to -102, and so on) result in negative Weighted Average Gains, inform the user that no change is needed.

            Think step by step. Answer in this format: "Reason:, Answer: new_att_rx"
            The "Answer" should be the new_att_rx value ONLY. If there is no change needed, write the current/ the old p0 nominal value only.
            **DO NOT use any tools**

            Here is a sample response:
            "There are currently 5 different values configured in the data available to me for att_rx. \
            Considering the KPIs resulting from these values, the optimal value is X. \
            There are two main KPIs used in the decision making process. The bitrate and SNR. \
            As the SNR increase causes UL interference, the tradeoff between SNR increase and \
            bitrate increase is calculated with Weighted average method and the best performing value compared to the current P0 nominal value, is -84."
            '''

            llm_response = llm_agent.invoke({"messages": prompt4})
            vars_new["att_rx"] = int(float(llm_response["messages"][-1].content.split('Answer: ')[1].split(',')[0]))
            return {"next": "Validation Agent", "agent_id": "config_agent", "messages": state["messages"]+[("assistant", llm_response["messages"][-1].content)], "average_kpis_df": average_kpis_df, "weighted_average_gain":weighted_avg, "vars_current": state["vars_current"], "vars_new": vars_new}

        except Exception as e:
            error_message = f"""Error: 🚨 Failed to extract new value of {param}. 
                LLM Raw Response: {llm_response["messages"][-1].content}
                Error: {str(e)}"""
            print(error_message)
            return {"next": None, "agent_id": "config_agent", "messages": state["messages"]+[("assistant", error_message)], "average_kpis_df": average_kpis_df, "weighted_average_gain":weighted_avg, "vars_current": state["vars_current"], "vars_new": None}

    else:
        error_message = f"""Error: 🚨 Failed to extract relevant parameter from query.
                Please make sure to restrict questions to permitted parameters only: p0_nominal, dl_carrierBandwidth, ul_carrierBandwidth, att_tx, att_rx


    Some examples of supported questions are: 

    What is the best value of p0 nominal? 
    Help me optimize dl carrierbandwidth.
    My current ul carrierbandwidth value is 51. Is there a better value for my network?
                """
        print(error_message)
        return {"next": None, "agent_id": "config_agent", "messages":  state["messages"]+[("assistant", error_message)], "average_kpis_df":None, "weighted_average_gain":None, "vars_current": state["vars_current"], "vars_new": None}

    print("Exiting Config Agent\n\n")
    return {"next": None, "agent_id": "config_agent", "messages": state["messages"], "average_kpis_df": average_kpis_df, "weighted_average_gain": weighted_average_gain, "vars_current": state["vars_current"], "vars_new": vars_new}


def valid_agent(state: State) -> State: 
    """
    5G Network Validation Agent for LangGraph.
    ------------------------------------------
    This agent validates the impact of a proposed network parameter change
    by applying the new configuration, collecting fresh KPIs, and computing 
    weighted performance improvements. If the change degrades performance, 
    the agent alerts the user and recommends changing the network to its previous state.

    The agent:
    1. Applies the new network parameter to gNodeB.
    2. Restarts the network and collects real-time KPIs.
    3. Generates SQL queries to aggregate KPI results.
    4. Calculates weighted average gain to assess benefit.
    5. Decides to keep or revert the change based on gain.
    """

    # Initialize LLM agent for validation tasks using config parameters
    llm = init_agent()
    system_prompt = '''You are an agent in a LangGraph. 
    Your task is to help an user validate and analyse a 5G network. You must reply to the questions asked concisely, and exactly in the format directed to you.'''
    llm_agent =  create_react_agent(llm, tools=[execute_xapp_sql, calc_weighted_average], prompt=system_prompt)
    
    # Persist the current network configuration to the database
    update_value_in_db(state["vars_current"])
    # Stop the network before applying a new configuration
    stop_network()

    # Identify which parameter has changed by comparing new and current values
    param = None
    for key in state["vars_new"].keys():
        if state["vars_new"][key]!= state["vars_current"][key]:
            param = key
            break
    
    # Define fallback procedure to safely revert to the previous stable configuration
    def fallback_to_prev_config(param):
        if check_network_status():
            stop_network()
        update_value_in_gnb(param, state["vars_current"][param])
        start_network()

    if param=="p0_nominal":
        try:
            # Apply the proposed parameter change to the network configuration
            update_value_in_gnb(param, state["vars_new"][param])
            start_network()
            yield f"✅ Network initiated with {param} = {state['vars_new'][param]}"
            
            # Wait for the configured monitoring duration to collect fresh KPI data
            validation_wait_time = yaml.safe_load(open('config.yaml', 'r'))['validation_wait_time']
            yield f"✅ Collecting KPIs for new {param} for {validation_wait_time} second(s)..."
            time.sleep(validation_wait_time)
            update_value_in_db(state["vars_new"])

            # Prompt the LLM to generate a SQL query for aggregating KPI data
            prompt2 = f''' Your task is to create a SQL query, execute it using the execute_xapp_sql tool EXACTLY ONCE, and return the resulting answer dataframe. \
            Do not make multiple calls to the tool. 
            There is a table called "{yaml.safe_load(open('config.yaml', 'r'))['table_name']}" with columns "tstamp", "pusch_snr", "p0_nominal", "dl_aggr_tbs".
            Write the SQL query which does the following:
            1. Makes another new column called "bitrate_dl", such that bitrate_dl[i] = max(0, (1000* (dl_aggr_tbs[i]-dl_aggr_tbs[i-1]))/(tstamp[i]-tstamp[i-1]))
            2. Makes a new column called "snr", such that snr[i] = pusch_snr[i]
            3. Finds the average of "bitrate_dl" and "snr" ,for these values of "p0_nominal" : {state["vars_new"]["p0_nominal"]}, {state["vars_current"]["p0_nominal"]}
            The SQL query should return me these columns in this order: "p0_nominal","Average_of_bitrate_dl", "Average_of_snr". 
            You should only print the dataframe, with column names, received from the execute_xapp_sql tool in following format: "data_frame : ,explanation: "'''

            llm_response = llm_agent.invoke({"messages": prompt2})
            average_kpis_df = llm_response["messages"][-2].content if hasattr(llm_response["messages"][-1], "content") else None
            print(average_kpis_df)
            yield "✅ Aggregrated data based on new KPI values"

        except Exception as e:
            error_message = (
                f"🚨 Error: Failed to configure and collect new KPIs.\n"
                f"Details: {str(e)}"
                )
            yield error_message
            yield "⚠️ Reverting to previous stable configuration..."
            fallback_to_prev_config(param)
            
            yield {"next": None, "agent_id": "valid_agent", "messages": state["messages"] + [("assistant", error_message)], "average_kpis_df": None, "weighted_average_gain": None, "vars_current": state["vars_current"], "vars_new": None}
            return 
        
        try:
            # Retrieve weighting factors used to calculate the weighted performance gain
            weight1, weight2 = yaml.safe_load(open('config.yaml', 'r'))[f'{param}_WA_weights']
            
            # Prompt the LLM to calculate weighted average gain using the collected data
            prompt3 = f''' Your task is to calculate the weighted average gain using the `calc_weighted_average` tool. The tool accepts the following parameters:
            Here are the inputs:
            1. `data_frame`: {average_kpis_df}
            2. `weight1`: {weight1}
            3. `weight2`: {weight2}
            4. `current_param_value`: {state["vars_current"]["p0_nominal"]}
            Use the tool to calculate the weighted average gain. Return **only** the resulting DataFrame as the output.'''

            llm_response = llm_agent.invoke({"messages":[prompt3]})
            weighted_avg = llm_response['messages'][-2].content
            print(weighted_avg)

            # Convert the result into a DataFrame and extract the numerical gain value
            weighted_avg_df = pd.read_csv(StringIO(weighted_avg), sep=r'\s+')
            weighted_avg_val = weighted_avg_df[weighted_avg_df.columns.tolist()[-1]].values[0]
            print(weighted_avg_val)

            yield f"✅ Weighted Average Gain observed: {weighted_avg_val:.2f}"
        
        except Exception as e:
            # Handle any failure gracefully and report the issue
            error_message = (
                f"🚨 Error: Failed to calculate weighted average calculation for {param}.\n"
                f"Details: {str(e)}"
                )
            yield error_message
            yield "⚠️ Reverting to previous stable configuration..."
            fallback_to_prev_config(param)
            yield {"next": None, "agent_id": "valid_agent", "messages": state["messages"] + [("assistant", error_message)], "average_kpis_df": None, "weighted_average_gain": None, "vars_current": state["vars_current"], "vars_new": None}
            return 
        
        try:
            # Evaluate if the weighted gain indicates an improvement
            if float(weighted_avg_val)<0:
                yield '''⚠️ Recommendation: Revert to Previous P0 Nominal Value Due to Reduced Weighted Gain '''
                update_value_in_db(state["vars_new"])
                yield "⚠️ Reverting to previous stable configuration..."
                fallback_to_prev_config(param)
                yield f"⚠️ {param} value updated to {state['vars_current'][param]} in gNodeB configuration file"
            else:
                state["vars_current"][param] = state["vars_new"][param]
                yield "✅ New configuration is optimal and stable."
                
        except Exception as e:

            error_message = f"🚨 Error: Failed to validate weighted average gain value. Reason: {str(e)}"
            yield error_message
            yield {"next": None, "agent_id": "valid_agent", "messages": state["messages"] + [("assistant", error_message)], "average_kpis_df": None, "weighted_average_gain": None, "vars_current": state["vars_current"], "vars_new": None}
            return 

    elif param=="dl_carrierBandwidth":
        try:
            # Apply the proposed parameter change to the network configuration
            update_value_in_gnb(param, state["vars_new"][param])
            state["vars_new"]["ul_carrierBandwidth"] = state["vars_new"]["dl_carrierBandwidth"]
            start_network()
            yield f"✅ Network initiated with {param} = {state['vars_new'][param]}"
            
            # Wait for the configured monitoring duration to collect fresh KPI data
            validation_wait_time = yaml.safe_load(open('config.yaml', 'r'))['validation_wait_time']
            yield f"✅ Collecting KPIs for new {param} for {validation_wait_time} second(s)..."
            time.sleep(validation_wait_time)
            update_value_in_db(state["vars_new"])
        
            # Prompt the LLM to generate a SQL query for aggregating KPI data
            prompt2 = f''' Your task is to create a SQL query, execute it using the execute_xapp_sql tool EXACTLY ONCE, and return the resulting answer dataframe. \
            Do not make multiple calls to the tool. 
            There is a table called "{yaml.safe_load(open('config.yaml', 'r'))['table_name']}" with columns "tstamp", "pusch_snr", "dl_carrierBandwidth", "dl_aggr_tbs".
            Write the SQL query which does the following:
            1. Makes another new column called "bitrate_dl", such that bitrate_dl[i] = max(0, (1000* (dl_aggr_tbs[i]-dl_aggr_tbs[i-1]))/(tstamp[i]-tstamp[i-1]))
            2. Makes a new column called "snr", such that snr[i] = pusch_snr[i]
            3. Finds the average of "bitrate_dl" and "snr" ,for these values of "dl_carrierBandwidth" : {state["vars_new"]["dl_carrierBandwidth"]}, {state["vars_current"]["dl_carrierBandwidth"]}
            The SQL query should return me these columns in this order: "dl_carrierBandwidth","Average_of_bitrate_dl", "Average_of_snr". 
            You should only print the dataframe, with column names, received from the execute_xapp_sql tool in following format: "data_frame : ,explanation: "'''

            llm_response = llm_agent.invoke({"messages": prompt2})
            average_kpis_df = llm_response["messages"][-2].content if hasattr(llm_response["messages"][-1], "content") else None
            print(average_kpis_df)
            yield "✅ Aggregrated data based on new KPI values"

        except Exception as e:
            error_message = (
                f"🚨 Error: Failed to configure and collect new KPIs.\n"
                f"Details: {str(e)}"
                )
            yield error_message
            yield "⚠️ Reverting to previous stable configuration..."
            fallback_to_prev_config(param)
            yield "⚠️ Reverting to previous stable configuration..."

            yield {"next": None, "agent_id": "valid_agent", "messages": state["messages"] + [("assistant", error_message)], "average_kpis_df": None, "weighted_average_gain": None, "vars_current": state["vars_current"], "vars_new": None}
            return 

        try:
            # Retrieve weighting factors used to calculate the weighted performance gain
            weight1, weight2 = yaml.safe_load(open('config.yaml', 'r'))[f'{param}_WA_weights']

            # Prompt the LLM to calculate weighted average gain using the collected data
            prompt3 = f''' Your task is to calculate the weighted average gain using the `calc_weighted_average` tool. The tool accepts the following parameters:
            Here are the inputs:
            1. `data_frame`: {average_kpis_df}
            2. `weight1`: {weight1}
            3. `weight2`: {weight2}
            4. `current_param_value`: {state["vars_current"]["dl_carrierBandwidth"]}
            Use the tool to calculate the weighted average gain. Return **only** the resulting DataFrame as the output.'''
            
            llm_response = llm_agent.invoke({"messages":[prompt3]})
            weighted_avg = llm_response['messages'][-2].content
            # print(llm_response)
            print(weighted_avg)

            # Convert the result into a DataFrame and extract the numerical gain value
            weighted_avg_df = pd.read_csv(StringIO(weighted_avg), sep=r'\s+')
            weighted_avg_val = weighted_avg_df[weighted_avg_df.columns.tolist()[-1]].values[0]
            print(weighted_avg_val)

            yield f"✅ Weighted Average Gain observed: {weighted_avg_val:.2f}"

        except Exception as e:
            # Handle any failure gracefully and report the issue
            error_message = (
                f"🚨 Error: Failed to calculate weighted average calculation for {param}.\n"
                f"Details: {str(e)}"
                )
            yield error_message
            yield {"next": None, "agent_id": "valid_agent", "messages": state["messages"] + [("assistant", error_message)], "average_kpis_df": None, "weighted_average_gain": None, "vars_current": state["vars_current"], "vars_new": None}
            return 

        try:
            # Evaluate if the weighted gain indicates an improvement
            if float(weighted_avg_val)<0:
                yield '''⚠️ Recommendation: Revert to Previous P0 Nominal Value Due to Reduced Weighted Gain '''
                update_value_in_db(state["vars_new"])
                yield "⚠️ Reverting to previous stable configuration..."
                fallback_to_prev_config(param)
                yield f"⚠️ {param} value updated to {state['vars_current'][param]} in gNodeB configuration file"
            else:
                state["vars_current"][param] = state["vars_new"][param]
                state["vars_current"]["ul_carrierBandwidth"] = state["vars_current"]["dl_carrierBandwidth"]
                yield "✅ New configuration is optimal and stable."
                
        except Exception as e:
            error_message = f"🚨 Error: Failed to validate weighted average gain value. Reason: {str(e)}"
            yield error_message
            yield {"next": None, "agent_id": "valid_agent", "messages": state["messages"] + [("assistant", error_message)], "average_kpis_df": None, "weighted_average_gain": None, "vars_current": state["vars_current"], "vars_new": None}
            return     
    
    elif param == "ul_carrierBandwidth":
        try:
            # Apply the proposed parameter change to the network configuration
            update_value_in_gnb(param, state["vars_new"][param])
            state["vars_new"]["dl_carrierBandwidth"] = state["vars_new"]["ul_carrierBandwidth"]
            start_network()
            yield f"✅ Network initiated with {param} = {state['vars_new'][param]}"
            
            # Wait for the configured monitoring duration to collect fresh KPI data
            validation_wait_time = yaml.safe_load(open('config.yaml', 'r'))['validation_wait_time']
            yield f"✅ Collecting KPIs for new {param} for {validation_wait_time} second(s)..."
            time.sleep(validation_wait_time)
            update_value_in_db(state["vars_new"])

            # Prompt the LLM to generate a SQL query for aggregating KPI data
            prompt2 = f''' Your task is to create a SQL query, execute it using the execute_xapp_sql tool EXACTLY ONCE, and return the resulting answer dataframe. \
            Do not make multiple calls to the tool. 
            There is a table called "{yaml.safe_load(open('config.yaml', 'r'))['table_name']}" with columns "tstamp", "pusch_snr", "ul_carrierBandwidth", "ul_aggr_tbs".
            Write the SQL query which does the following:
            1. Makes another new column called "bitrate_ul", such that bitrate_ul[i] = max(0, (1000* (ul_aggr_tbs[i]-ul_aggr_tbs[i-1]))/(tstamp[i]-tstamp[i-1]))
            2. Makes a new column called "snr", such that snr[i] = pusch_snr[i]
            3. Finds the average of "bitrate_ul" and "snr" ,for these values of "ul_carrierBandwidth" : {state["vars_new"]["ul_carrierBandwidth"]}, {state["vars_current"]["ul_carrierBandwidth"]}
            The SQL query should return me these columns in this order: "ul_carrierBandwidth","Average_of_bitrate_ul", "Average_of_snr". 
            You should only print the dataframe, with column names, received from the execute_xapp_sql tool in following format: "data_frame : ,explanation: "'''


            llm_response = llm_agent.invoke({"messages": prompt2})
            average_kpis_df = llm_response["messages"][-2].content if hasattr(llm_response["messages"][-1], "content") else None
            print(average_kpis_df)
            yield "✅ Aggregrated data based on new KPI values"
        except Exception as e:
            error_message = (
                f"🚨 Error: Failed to configure and collect new KPIs.\n"
                f"Details: {str(e)}"
                )

            yield error_message
            yield "⚠️ Reverting to previous stable configuration..."
            fallback_to_prev_config(param)
            yield {"next": None, "agent_id": "valid_agent", "messages": state["messages"] + [("assistant", error_message)], "average_kpis_df": None, "weighted_average_gain": None, "vars_current": state["vars_current"], "vars_new": None}
            return 

        try:
            # Retrieve weighting factors used to calculate the weighted performance gain
            weight1, weight2 = yaml.safe_load(open('config.yaml', 'r'))[f'{param}_WA_weights']
            
            # Prompt the LLM to calculate weighted average gain using the collected data
            prompt3 = f''' Your task is to calculate the weighted average gain using the `calc_weighted_average` tool. The tool accepts the following parameters:
            Here are the inputs:
            1. `data_frame`: 
            
            {average_kpis_df}
            2. `weight1`: {weight1}
            3. `weight2`: {weight2}
            4. `current_param_value`: {state["vars_current"]["ul_carrierBandwidth"]}
            Use the tool to calculate the weighted average gain. Return **only** the resulting DataFrame as the output.'''
            
            llm_response = llm_agent.invoke({"messages":[prompt3]})
            weighted_avg = llm_response['messages'][-2].content
            # print(llm_response)
            print(weighted_avg)

            # Convert the result into a DataFrame and extract the numerical gain value
            weighted_avg_df = pd.read_csv(StringIO(weighted_avg), sep=r'\s+')
            weighted_avg_val = weighted_avg_df[weighted_avg_df.columns.tolist()[-1]].values[0]
            print(weighted_avg_val)

            yield f"✅ Weighted Average Gain observed: {weighted_avg_val:.2f}"

        except Exception as e:
            # Handle any failure gracefully and report the issue
            error_message = (
                f"🚨 Error: Failed to calculate weighted average calculation for {param}.\n"
                f"Details: {str(e)}"
                )
            yield error_message
            yield {"next": None, "agent_id": "valid_agent", "messages": state["messages"] + [("assistant", error_message)], "average_kpis_df": None, "weighted_average_gain": None, "vars_current": state["vars_current"], "vars_new": None}
            return 
        
        try:
            # Evaluate if the weighted gain indicates an improvement
            if float(weighted_avg_val)<0:
                yield '''⚠️ Recommendation: Revert to Previous P0 Nominal Value Due to Reduced Weighted Gain '''
                update_value_in_db(state["vars_new"])

                yield "⚠️ Reverting to previous stable configuration..."        
                fallback_to_prev_config(param)
                yield f"⚠️ {param} value updated to {state['vars_current'][param]} in gNodeB configuration file"
            else:
                state["vars_current"][param] = state["vars_new"][param]
                state["vars_current"]["dl_carrierBandwidth"] = state["vars_current"]["ul_carrierBandwidth"]
                yield "✅ New configuration is optimal and stable."
                
        except Exception as e:
            error_message = f"🚨 Error: Failed to validate weighted average gain value. Reason: {str(e)}"
            yield error_message
            yield {"next": None, "agent_id": "valid_agent", "messages": state["messages"] + [("assistant", error_message)], "average_kpis_df": None, "weighted_average_gain": None, "vars_current": state["vars_current"], "vars_new": None}
            return  
    
    
    elif param == "att_tx":
        try:
            # Apply the proposed parameter change to the network configuration
            update_value_in_gnb(param, state["vars_new"][param])
            start_network()
            yield f"✅ Network initiated with {param} = {state['vars_new'][param]}"

            # Wait for the configured monitoring duration to collect fresh KPI data
            validation_wait_time = yaml.safe_load(open('config.yaml', 'r'))['validation_wait_time']
            yield f"✅ Collecting KPIs for new {param} for {validation_wait_time} second(s)..."
            time.sleep(validation_wait_time)
            update_value_in_db(state["vars_new"])

            # Prompt the LLM to generate a SQL query for aggregating KPI data
            prompt2 = f''' Your task is to create a SQL query, execute it using the execute_xapp_sql tool EXACTLY ONCE, and return the resulting answer dataframe. \
            Do not make multiple calls to the tool. 
            There is a table called "{yaml.safe_load(open('config.yaml', 'r'))['table_name']}" with columns "tstamp", "pusch_snr", "att_tx", "dl_aggr_tbs".
            Write the SQL query which does the following:
            1. Makes another new column called "bitrate_dl", such that bitrate_dl[i] = max(0, (1000* (dl_aggr_tbs[i]-dl_aggr_tbs[i-1]))/(tstamp[i]-tstamp[i-1]))
            2. Makes a new column called "snr", such that snr[i] = pusch_snr[i]
            3. Finds the average of "bitrate_dl" and "snr" ,for these values of "att_tx" : {state["vars_new"]["att_tx"]}, {state["vars_current"]["att_tx"]}
            The SQL query should return me these columns in this order: "att_tx","Average_of_bitrate_dl", "Average_of_snr". 
            You should only print the dataframe, with column names, received from the execute_xapp_sql tool in following format: "data_frame : ,explanation: "'''

            llm_response = llm_agent.invoke({"messages": prompt2})
            average_kpis_df = llm_response["messages"][-2].content if hasattr(llm_response["messages"][-1], "content") else None
            print(average_kpis_df)
            yield "✅ Aggregrated data based on new KPI values"

        except Exception as e:
            error_message = (
                f"🚨 Error: Failed to configure and collect new KPIs.\n"
                f"Details: {str(e)}"
                )
            yield error_message
            yield "⚠️ Reverting to previous stable configuration..."
            fallback_to_prev_config(param)
            yield {"next": None, "agent_id": "valid_agent", "messages": state["messages"] + [("assistant", error_message)], "average_kpis_df": None, "weighted_average_gain": None, "vars_current": state["vars_current"], "vars_new": None}
            return 
        
        try:
            # Retrieve weighting factors used to calculate the weighted performance gain
            weight1, weight2 = yaml.safe_load(open('config.yaml', 'r'))[f'{param}_WA_weights']

            # Prompt the LLM to calculate weighted average gain using the collected data
            prompt3 = f''' Your task is to calculate the weighted average gain using the `calc_weighted_average` tool. The tool accepts the following parameters:
            Here are the inputs:
            1. `data_frame`: 
            {average_kpis_df}
            2. `weight1`: {weight1}
            3. `weight2`: {weight2}
            4. `current_param_value`: {state["vars_current"]["att_tx"]}
            Use the tool to calculate the weighted average gain. Return **only** the resulting DataFrame as the output.'''
            llm_response = llm_agent.invoke({"messages":[prompt3]})
            weighted_avg = llm_response['messages'][-2].content
            print(weighted_avg)

            # Convert the result into a DataFrame and extract the numerical gain value
            weighted_avg_df = pd.read_csv(StringIO(weighted_avg), sep=r'\s+')
            weighted_avg_val = weighted_avg_df[weighted_avg_df.columns.tolist()[-1]].values[0]
            print(weighted_avg_val)
            yield f"✅ Weighted Average Gain observed: {weighted_avg_val:.2f}"

        except Exception as e:
            # Handle any failure gracefully and report the issue
            error_message = (
                f"🚨 Error: Failed to calculate weighted average calculation for {param}.\n"
                f"Details: {str(e)}"
                )
            yield error_message
            yield {"next": None, "agent_id": "valid_agent", "messages": state["messages"] + [("assistant", error_message)], "average_kpis_df": None, "weighted_average_gain": None, "vars_current": state["vars_current"], "vars_new": None}
            return 

        try:
            # Evaluate if the weighted gain indicates an improvement
            if float(weighted_avg_val)<0:
                yield '''⚠️ Recommendation: Revert to Previous P0 Nominal Value Due to Reduced Weighted Gain '''
                update_value_in_db(state["vars_new"])

                yield error_message
                yield "⚠️ Reverting to previous stable configuration..."
                fallback_to_prev_config(param)
                yield f"⚠️ {param} value updated to {state['vars_current'][param]} in gNodeB configuration file"

            else:
                state["vars_current"][param] = state["vars_new"][param]
                yield "✅ New configuration is optimal and stable."
                
        except Exception as e:

            error_message = f"🚨 Error: Failed to validate weighted average gain value. Reason: {str(e)}"
            yield error_message
            yield {"next": None, "agent_id": "valid_agent", "messages": state["messages"] + [("assistant", error_message)], "average_kpis_df": None, "weighted_average_gain": None, "vars_current": state["vars_current"], "vars_new": None}
            return 
    
    
    elif param == "att_rx":
        try:
            # Apply the proposed parameter change to the network configuration
            update_value_in_gnb(param, state["vars_new"][param])
            start_network()
            yield f"✅ Network initiated with {param} = {state['vars_new'][param]}"

            # Wait for the configured monitoring duration to collect fresh KPI data
            validation_wait_time = yaml.safe_load(open('config.yaml', 'r'))['validation_wait_time']
            yield f"✅ Collecting KPIs for new {param} for {validation_wait_time} second(s)..."
            time.sleep(validation_wait_time)
            update_value_in_db(state["vars_new"])
        
            # Prompt the LLM to generate a SQL query for aggregating KPI data
            prompt2 = f''' Your task is to create a SQL query, execute it using the execute_xapp_sql tool EXACTLY ONCE, and return the resulting answer dataframe. \
            Do not make multiple calls to the tool. 
            There is a table called "{yaml.safe_load(open('config.yaml', 'r'))['table_name']}" with columns "tstamp", "dl_harq_round0", "dl_harq_round1", "dl_harq_round2", "dl_harq_round3", "att_rx", "ul_aggr_tbs".
            Write the SQL query which does the following:
            1. Makes another new column called "bitrate_ul", such that bitrate_ul[i] = max(0, (1000* (ul_aggr_tbs[i]-ul_aggr_tbs[i-1]))/(tstamp[i]-tstamp[i-1]))
            2.  Makes a new column called "retx", such that retx[i] = (1000*(dl_harq_round0[i] + dl_harq_round1[i] + dl_harq_round2[i] + dl_harq_round3[i] - dl_harq_round0[i-1] - dl_harq_round1[i-1] - dl_harq_round2[i-1] - dl_harq_round3[i-1]) )/ (tstamp[i]-tstamp[i-1])
            3. Finds the average of "bitrate_ul" and "retx" ,for these values of "att_rx" : {state["vars_new"]["att_rx"]}, {state["vars_current"]["att_rx"]}
            The SQL query should return me these columns in this order: "att_rx","Average_of_bitrate_ul", "Average_of_retx". 
            You should only print the dataframe, with column names, received from the execute_xapp_sql tool in following format: "data_frame : ,explanation: "'''

            llm_response = llm_agent.invoke({"messages": prompt2})
            average_kpis_df = llm_response["messages"][-2].content if hasattr(llm_response["messages"][-1], "content") else None
            print(average_kpis_df)
            yield "✅ Aggregrated data based on new KPI values"

        except Exception as e:
            error_message = (
                f"🚨 Error: Failed to configure and collect new KPIs.\n"
                f"Details: {str(e)}"
                )

            yield error_message
            yield "⚠️ Reverting to previous stable configuration..."
            fallback_to_prev_config(param)
            yield {"next": None, "agent_id": "valid_agent", "messages": state["messages"] + [("assistant", error_message)], "average_kpis_df": None, "weighted_average_gain": None, "vars_current": state["vars_current"], "vars_new": None}
            return 

        try:
            # Retrieve weighting factors used to calculate the weighted performance gain
            weight1, weight2 = yaml.safe_load(open('config.yaml', 'r'))[f'{param}_WA_weights']

            # Prompt the LLM to calculate weighted average gain using the collected data
            prompt3 = f''' Your task is to calculate the weighted average gain using the `calc_weighted_average` tool. The tool accepts the following parameters:
            Here are the inputs:
            1. `data_frame`: {average_kpis_df}
            2. `weight1`: {weight1}
            3. `weight2`: {weight2}
            4. `current_param_value`: {state["vars_current"]["att_rx"]}
            Use the tool to calculate the weighted average gain. Return **only** the resulting DataFrame as the output.'''
            
            llm_response = llm_agent.invoke({"messages":[prompt3]})
            weighted_avg = llm_response['messages'][-2].content

            # Convert the result into a DataFrame and extract the numerical gain value
            weighted_avg_df = pd.read_csv(StringIO(weighted_avg), sep=r'\s+')
            weighted_avg_val = weighted_avg_df[weighted_avg_df.columns.tolist()[-1]].values[0]
            print(weighted_avg_val)
            yield f"✅ Weighted Average Gain observed: {weighted_avg_val:.2f}"

        except Exception as e:
            # Handle any failure gracefully and report the issue
            error_message = (
                f"🚨 Error: Failed to calculate weighted average calculation for {param}.\n"
                f"Details: {str(e)}"
                )
            yield error_message
            yield {"next": None, "agent_id": "valid_agent", "messages": state["messages"] + [("assistant", error_message)], "average_kpis_df": None, "weighted_average_gain": None, "vars_current": state["vars_current"], "vars_new": None}
            return 

        try:
            if float(weighted_avg_val)<0:
                yield '''⚠️ Recommendation: Revert to Previous P0 Nominal Value Due to Reduced Weighted Gain '''
                update_value_in_db(state["vars_new"])
                yield "⚠️ Reverting to previous stable configuration..."
                fallback_to_prev_config(param)
                yield f"⚠️ {param} value updated to {state['vars_current'][param]} in gNodeB configuration file"
            else:
                state["vars_current"][param] = state["vars_new"][param]
                yield "✅ New configuration is optimal and stable."
                
        except Exception as e:
            # Evaluate if the weighted gain indicates an improvement
            error_message = f"🚨 Error: Failed to validate weighted average gain value. Reason: {str(e)}"
            yield error_message
            yield {"next": None, "agent_id": "valid_agent", "messages": state["messages"] + [("assistant", error_message)], "average_kpis_df": None, "weighted_average_gain": None, "vars_current": state["vars_current"], "vars_new": None}
            return     
            
    else:
        error_message = "🚨 Error: No parameter validation detected."
        print(error_message)
        yield {"next": None, "agent_id": "valid_agent", "messages": state["messages"] + [("assistant", error_message)], "average_kpis_df": None, "weighted_average_gain": None, "vars_current": state["vars_current"], "vars_new": None}
        return 

    yield {"next": None, "agent_id": "valid_agent", "messages": state["messages"], "average_kpis_df":average_kpis_df, "weighted_average_gain":weighted_avg_df, "vars_current": state["vars_current"], "vars_new": None}
    return

# Check whether NIM access is verified and if there are any issues accessing endpoints
def test_NIM():
    """
    Test the connectivity and readiness of a locally hosted NIM.
    This includes:
        - Checking if the NIM health endpoint is accessible.
        - Verifying the required model exists in the model list.
    Returns:
        str: A success message or detailed error message depending on test outcomes.
    """
    # Load details from configuration file
    nim_llm_port = yaml.safe_load(open('config.yaml', 'r'))['nim_llm_port']

    if "BUBBLERAN_HOST_PWD" in os.environ:
        nim_base_url = f"http://host.docker.internal:{nim_llm_port}/v1"
    else:
        nim_base_url = f"http://localhost:{nim_llm_port}/v1"

    success_message = "Success: Access to locally hosted NIM verified."
    error_message = "Error encountered while testing access to local NIMs. Ensure you have NIMs deployed, and correct details in config.yaml file."
    
    # Check NIM health status
    try:
        response = requests.get(nim_base_url+"/health/ready")
        if response.status_code == 200:
            print("NIM is healthy and ready!")
        else:
            print(f"NIM is not ready. Status code: {response.status_code}")
            return error_message + "Status code: " + response.status_code
    except Exception as e:
        print(f"Failed to connect to NIM service at {nim_base_url}: {e}")
        print(f"{error_message} Details: {e}")
        return f"{error_message} Details: {e}"
    
    # Check if specified model is available
    try:
        nim_image = yaml.safe_load(open('config.yaml', 'r'))['nim_image']
        response = requests.get(f"{nim_base_url}/models")
        if response.status_code == 200:
            models_json = response.json()  # Expected structure: {"data": [{"id": ...}, ...]}
            model_ids = [model["id"] for model in models_json.get("data", [])]
            # Extract model name from image string
            nim_llm_model = nim_image.split("nvcr.io/nim/")[-1].split(":")[0]
            if nim_llm_model not in model_ids:
                error_message = "Error in accessing NIM image specified in config.yaml. Please check the details NIM image is correct and in specified format."
                print(error_message)
                print("Available models:", model_ids)
                return error_message
        else:
            print(f"{error_message} Failed to get models list. Status code: {response.status_code}")
            return f"{error_message} Failed to get models list. Status code: {response.status_code}"
    except Exception as e:
        print(f"{error_message} Details: {e}")
        return f"{error_message} Details: {e}"

    return success_message   

# Run a standalone sanity check if the script is executed directly
if __name__=="__main__":
    # Initialize the LLM agent with a basic system prompt
    test_NIM()

