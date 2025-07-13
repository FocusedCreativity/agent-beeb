Application Structure¶
Overview¶
A LangGraph application consists of one or more graphs, a configuration file (langgraph.json), a file that specifies dependencies, and an optional .env file that specifies environment variables.

This guide shows a typical structure of an application and shows how the required information to deploy an application using the LangGraph Platform is specified.

Key Concepts¶
To deploy using the LangGraph Platform, the following information should be provided:

A LangGraph configuration file (langgraph.json) that specifies the dependencies, graphs, and environment variables to use for the application.
The graphs that implement the logic of the application.
A file that specifies dependencies required to run the application.
Environment variables that are required for the application to run.
File Structure¶
Below are examples of directory structures for Python and JavaScript applications:


Python (requirements.txt)
Python (pyproject.toml)
JS (package.json)

my-app/
├── my_agent # all project code lies within here
│   ├── utils # utilities for your graph
│   │   ├── __init__.py
│   │   ├── tools.py # tools for your graph
│   │   ├── nodes.py # node functions for your graph
│   │   └── state.py # state definition of your graph
│   ├── __init__.py
│   └── agent.py # code for constructing your graph
├── .env # environment variables
├── requirements.txt # package dependencies
└── langgraph.json # configuration file for LangGraph

Note

The directory structure of a LangGraph application can vary depending on the programming language and the package manager used.

Configuration File¶
The langgraph.json file is a JSON file that specifies the dependencies, graphs, environment variables, and other settings required to deploy a LangGraph application.

See the LangGraph configuration file reference for details on all supported keys in the JSON file.

Tip

The LangGraph CLI defaults to using the configuration file langgraph.json in the current directory.

Examples¶

Python
JavaScript
The dependencies involve a custom local package and the langchain_openai package.
A single graph will be loaded from the file ./your_package/your_file.py with the variable variable.
The environment variables are loaded from the .env file.

{
    "dependencies": [
        "langchain_openai",
        "./your_package"
    ],
    "graphs": {
        "my_agent": "./your_package/your_file.py:agent"
    },
    "env": "./.env"
}

Dependencies¶
A LangGraph application may depend on other Python packages or JavaScript libraries (depending on the programming language in which the application is written).

You will generally need to specify the following information for dependencies to be set up correctly:

A file in the directory that specifies the dependencies (e.g. requirements.txt, pyproject.toml, or package.json).
A dependencies key in the LangGraph configuration file that specifies the dependencies required to run the LangGraph application.
Any additional binaries or system libraries can be specified using dockerfile_lines key in the LangGraph configuration file.
Graphs¶
Use the graphs key in the LangGraph configuration file to specify which graphs will be available in the deployed LangGraph application.

You can specify one or more graphs in the configuration file. Each graph is identified by a name (which should be unique) and a path for either: (1) the compiled graph or (2) a function that makes a graph is defined.

Environment Variables¶
If you're working with a deployed LangGraph application locally, you can configure environment variables in the env key of the LangGraph configuration file.

For a production deployment, you will typically want to configure the environment variables in the deployment environment.