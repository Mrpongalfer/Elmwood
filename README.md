# Elmwood
Elmwood TUI with agent exwork

Project Structure
First, create the following directory structure. You will place the code for each file I provide into its corresponding location.
/sentient_tui/
├── terminus/                # The Rust runtime/orchestrator
│   ├── src/
│   │   ├── main.rs
│   │   ├── vdom.rs          # Terminal Virtual DOM structures and diffing logic
│   │   ├── js_runtime.rs    # Logic for embedding the JS engine and running Elm
│   │   └── terminal_io.rs   # Handles crossterm events and rendering
│   └── Cargo.toml
├── elmwood/                 # The Elm TUI application
│   ├── src/
│   │   ├── Main.elm
│   │   └── Elmwood/
│   │       ├── VDom.elm
│   │       ├── Events.elm
│   │       └── Attributes.elm
│   └── elm.json
├── exwork/                  # The Python Agent
│   ├── agent_ex_work.py
│   └── requirements.txt
└── README.md                # Build and run instructions


Part 1: Agent Ex-Work (The Python Effector)
This is the complete, standalone Python agent, built according to the specifications you provided.
exwork/requirements.txt
python-dotenv
requests


exwork/agent_ex_work.py
#!/usr/bin/env python3
import os
import sys
import json
import subprocess
import base64
import time
import logging
from functools import wraps
from pathlib import Path
from typing import Dict, Any, List, Callable, Optional
from dotenv import load_dotenv
import requests

# --- Configuration ---
load_dotenv()
PROJECT_ROOT = Path(os.getcwd())
HISTORY_FILE = PROJECT_ROOT / ".exwork_history.jsonl"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

# Load from environment with defaults
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")

# --- Logging Setup ---
logging.basicConfig(
    filename=PROJECT_ROOT / ".exwork_agent.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# --- Handler Registry ---
HANDLERS: Dict[str, Callable] = {}

def handler(name: str) -> Callable:
    """Decorator to register a function as an action handler."""
    def decorator(func: Callable) -> Callable:
        HANDLERS[name] = func
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    return decorator

# --- Utility Functions ---
def resolve_path(file_path: str) -> Optional[Path]:
    """
    Safely resolves a path, ensuring it's within the project root.
    Prevents directory traversal attacks.
    """
    try:
        # Normalize the path to resolve '..' etc.
        resolved_path = (PROJECT_ROOT / file_path).resolve()
        # Check if the resolved path is within the project root
        if PROJECT_ROOT in resolved_path.parents or resolved_path == PROJECT_ROOT or resolved_path.parent == PROJECT_ROOT:
             if resolved_path.is_relative_to(PROJECT_ROOT):
                return resolved_path
    except Exception as e:
        logging.error(f"Path resolution error for '{file_path}': {e}")
    return None

def log_execution_history(action: Dict, result: Dict):
    """Logs the details of an executed action to the history file."""
    log_entry = {
        "timestamp": time.time(),
        "action_type": action.get("type"),
        "parameters": action.get("parameters"),
        "result": {
            "success": result.get("success"),
            "status": result.get("status"),
            "duration_seconds": result.get("duration_seconds"),
            "stdout": result.get("details", {}).get("stdout"),
            "stderr": result.get("details", {}).get("stderr"),
        }
    }
    try:
        with open(HISTORY_FILE, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
    except Exception as e:
        logging.error(f"Failed to write to execution history: {e}")

# --- Core Action Handlers ---
@handler("ECHO")
def echo_handler(params: Dict) -> Dict:
    message = params.get("message", "")
    print(message, file=sys.stderr) # Print to stderr to not interfere with stdout JSON
    return {"success": True, "status": "Message echoed."}

@handler("CREATE_OR_REPLACE_FILE")
def create_or_replace_file_handler(params: Dict) -> Dict:
    file_path_str = params.get("file_path")
    content_b64 = params.get("content_b64")
    if not file_path_str or content_b64 is None:
        return {"success": False, "status": "Missing file_path or content_b64."}
    
    file_path = resolve_path(file_path_str)
    if not file_path:
        return {"success": False, "status": f"Invalid or unsafe path: {file_path_str}"}

    try:
        content_bytes = base64.b64decode(content_b64)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "wb") as f:
            f.write(content_bytes)
        return {"success": True, "status": f"File '{file_path_str}' created/replaced."}
    except Exception as e:
        return {"success": False, "status": f"Failed to write file: {e}"}

@handler("RUN_SCRIPT")
def run_script_handler(params: Dict) -> Dict:
    command_list = params.get("command")
    if not command_list or not isinstance(command_list, list):
        return {"success": False, "status": "Invalid 'command' parameter, must be a list."}
    
    timeout = params.get("timeout", 60)
    script_path = resolve_path(command_list[0])

    # For security, only allow scripts in project root or ./scripts
    if not script_path or not (script_path.parent == PROJECT_ROOT or script_path.parent == SCRIPTS_DIR):
        # Allow system commands if the path is not a file path
        if not Path(command_list[0]).is_file():
             pass # Assume it's a system command like 'git' or 'ruff'
        else:
             return {"success": False, "status": "Scripts can only be run from project root or ./scripts subdir."}

    try:
        if script_path and script_path.is_file():
            os.chmod(script_path, 0o755) # Ensure script is executable

        process = subprocess.run(
            command_list,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=PROJECT_ROOT,
        )
        success = process.returncode == 0
        return {
            "success": success,
            "status": "Command finished." if success else "Command failed.",
            "details": {
                "exit_code": process.returncode,
                "stdout": process.stdout,
                "stderr": process.stderr,
            },
        }
    except FileNotFoundError:
        return {"success": False, "status": f"Command not found: {command_list[0]}"}
    except subprocess.TimeoutExpired:
        return {"success": False, "status": f"Command timed out after {timeout} seconds."}
    except Exception as e:
        return {"success": False, "status": f"An error occurred: {e}"}

@handler("CALL_LOCAL_LLM")
def call_local_llm_handler(params: Dict) -> Dict:
    prompt = params.get("prompt")
    if not prompt:
        return {"success": False, "status": "Missing prompt."}
    
    api_url = f"{OLLAMA_HOST}/api/generate"
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False
    }
    try:
        response = requests.post(api_url, json=payload, timeout=120)
        response.raise_for_status()
        
        data = response.json()
        return {
            "success": True,
            "status": "LLM call successful.",
            "details": {
                "response": data.get("response")
            }
        }
    except Exception as e:
        return {"success": False, "status": f"Failed to call local LLM: {e}"}

# --- Main Execution Logic ---
def execute_action(action: Dict) -> Dict:
    """Looks up and executes a single action by its type."""
    action_type = action.get("type")
    params = action.get("parameters", {})
    
    handler_func = HANDLERS.get(action_type)
    if not handler_func:
        return {"success": False, "status": f"Unknown action type: {action_type}"}

    start_time = time.time()
    try:
        result = handler_func(params)
    except Exception as e:
        logging.error(f"Handler for '{action_type}' crashed: {e}")
        result = {"success": False, "status": f"Internal error in handler: {e}"}
    
    duration = time.time() - start_time
    result["duration_seconds"] = round(duration, 4)
    result["action_type"] = action_type
    
    log_execution_history(action, result)
    return result


def main():
    """Main entry point. Reads a JSON instruction block from stdin and executes it."""
    try:
        input_json_str = sys.stdin.read()
        if not input_json_str:
            sys.exit("Ex-Work: No input received on stdin.")

        instruction_block = json.loads(input_json_str)
    except json.JSONDecodeError:
        result = {
            "overall_success": False,
            "status_message": "Failed to decode input JSON.",
            "action_results": []
        }
        print(json.dumps(result))
        sys.exit(1)

    start_time = time.time()
    actions = instruction_block.get("actions", [])
    action_results = []
    overall_success = True

    for action in actions:
        result = execute_action(action)
        action_results.append(result)
        if not result["success"]:
            overall_success = False
            # Option to stop on first failure can be added here
            # break 

    total_duration = time.time() - start_time

    final_output = {
        "step_id": instruction_block.get("step_id"),
        "overall_success": overall_success,
        "status_message": "All actions completed." if overall_success else "One or more actions failed.",
        "duration_seconds": round(total_duration, 4),
        "action_results": action_results,
    }

    print(json.dumps(final_output, indent=2))

if __name__ == "__main__":
    main()


Part 2: Elmwood (The Elm TUI Core)
This is the pure functional core of our TUI application.
elmwood/elm.json
{
    "type": "application",
    "source-directories": [
        "src"
    ],
    "elm-version": "0.19.1",
    "dependencies": {
        "direct": {
            "elm/browser": "1.0.2",
            "elm/core": "1.0.5",
            "elm/html": "1.0.0",
            "elm/json": "1.1.3"
        },
        "indirect": {
            "elm/time": "1.0.0",
            "elm/url": "1.0.0",
            "elm/virtual-dom": "1.0.3"
        }
    },
    "test-dependencies": {
        "direct": {},
        "indirect": {}
    }
}


elmwood/src/Elmwood/VDom.elm
module Elmwood.VDom exposing (..)

import Elmwood.Attributes exposing (Attribute, mapAttributes)
import Json.Encode as E


type VDom msg
    = VText (List (Attribute msg)) String
    | VBox (List (Attribute msg)) (List (VDom msg))
    | HStack (List (Attribute msg)) (List (VDom msg))
    | VStack (List (Attribute msg)) (List (VDom msg))
    | Separator (List (Attribute msg))


type alias Document msg =
    VDom msg


document : List (Attribute msg) -> List (VDom msg) -> Document msg
document =
    VBox


text : String -> VDom msg
text content =
    VText [] content


box : List (Attribute msg) -> List (VDom msg) -> VDom msg
box =
    VBox


hstack : List (Attribute msg) -> List (VDom msg) -> VDom msg
hstack =
    HStack


vstack : List (Attribute msg) -> List (VDom msg) -> VDom msg
vstack =
    VStack


separator : VDom msg
separator =
    Separator []


map : (a -> b) -> VDom a -> VDom b
map fn vdom =
    case vdom of
        VText attrs str ->
            VText (mapAttributes fn attrs) str

        VBox attrs children ->
            VBox (mapAttributes fn attrs) (List.map (map fn) children)

        HStack attrs children ->
            HStack (mapAttributes fn attrs) (List.map (map fn) children)

        VStack attrs children ->
            VStack (mapAttributes fn attrs) (List.map (map fn) children)

        Separator attrs ->
            Separator (mapAttributes fn attrs)


encode : VDom msg -> E.Value
encode vdom =
    case vdom of
        VText attrs str ->
            E.object
                [ ( "type", E.string "Text" )
                , ( "attributes", Attributes.encode attrs )
                , ( "content", E.string str )
                ]

        VBox attrs children ->
            encodeContainer "VBox" attrs children

        HStack attrs children ->
            encodeContainer "HStack" attrs children

        VStack attrs children ->
            encodeContainer "VStack" attrs children

        Separator attrs ->
            E.object
                [ ( "type", E.string "Separator" )
                , ( "attributes", Attributes.encode attrs )
                ]


encodeContainer : String -> List (Attribute msg) -> List (VDom msg) -> E.Value
encodeContainer typeStr attrs children =
    E.object
        [ ( "type", E.string typeStr )
        , ( "attributes", Attributes.encode attrs )
        , ( "children", E.list encode children )
        ]


elmwood/src/Elmwood/Attributes.elm
module Elmwood.Attributes exposing (..)

import Json.Encode as E


type Attribute msg
    = AlignCenter
    | JustifyCenter
    | Padding Int
    | Bold
    | FgColor String
    | BorderRounded
    | OnKeyPress (String -> msg)


alignCenter : Attribute msg
alignCenter =
    AlignCenter


justifyCenter : Attribute msg
justifyCenter =
    JustifyCenter


padding : Int -> Attribute msg
padding =
    Padding


bold : Attribute msg
bold =
    Bold


fgColor : String -> Attribute msg
fgColor =
    FgColor


borderRounded : Attribute msg
borderRounded =
    BorderRounded


mapAttributes : (a -> b) -> List (Attribute a) -> List (Attribute b)
mapAttributes fn attrs =
    List.map
        (\attr ->
            case attr of
                OnKeyPress handler ->
                    OnKeyPress (handler >> fn)

                AlignCenter ->
                    AlignCenter

                JustifyCenter ->
                    JustifyCenter

                Padding p ->
                    Padding p

                Bold ->
                    Bold

                FgColor c ->
                    FgColor c

                BorderRounded ->
                    BorderRounded
        )
        attrs


encode : List (Attribute msg) -> E.Value
encode attrs =
    let
        encodeAttr attr =
            case attr of
                AlignCenter ->
                    E.object [ ( "type", E.string "AlignCenter" ) ]

                JustifyCenter ->
                    E.object [ ( "type", E.string "JustifyCenter" ) ]

                Padding p ->
                    E.object [ ( "type", E.string "Padding" ), ( "value", E.int p ) ]

                Bold ->
                    E.object [ ( "type", E.string "Bold" ) ]

                FgColor c ->
                    E.object [ ( "type", E.string "FgColor" ), ( "value", E.string c ) ]

                BorderRounded ->
                    E.object [ ( "type", E.string "BorderRounded" ) ]

                OnKeyPress _ ->
                    -- Handlers are not encoded, they are managed by the runtime
                    E.null
    in
    attrs
        |> List.map encodeAttr
        |> E.list


elmwood/src/Elmwood/Events.elm
module Elmwood.Events exposing (onKeyPress)

import Elmwood.Attributes exposing (Attribute(..))


onKeyPress : (String -> msg) -> Attribute msg
onKeyPress =
    OnKeyPress


elmwood/src/Main.elm
port module Main exposing (main)

import Browser
import Elmwood.Attributes as Attrs
import Elmwood.Events exposing (onKeyPress)
import Elmwood.VDom as V
import Json.Decode as D
import Json.Encode as E


-- PORTS: The bridge between Elm (pure) and Rust (impure)
port executeCommand : E.Value -> Cmd msg
port terminalEvents : (D.Value -> msg) -> Sub msg


-- MODEL
type alias Model =
    { status : String
    , lastResult : Maybe String
    , currentLog : List String
    }


init : ( Model, Cmd Msg )
init =
    ( { status = "INITIALIZING...", lastResult = Nothing, currentLog = [] }
    , Cmd.none
    )


-- MSG
type Msg
    = NoOp
    | TerminalEvent (Result D.Error D.Value)
    | KeyPressed String
    | ExWorkResult (Result D.Error ExWorkResult)
    | SendEchoCommand


type alias ExWorkResult =
    { overall_success : Bool
    , status_message : String
    }


exWorkResultDecoder : D.Decoder ExWorkResult
exWorkResultDecoder =
    D.map2 ExWorkResult
        (D.field "overall_success" D.bool)
        (D.field "status_message" D.string)


-- UPDATE
update : Msg -> Model -> ( Model, Cmd Msg )
update msg model =
    case msg of
        NoOp ->
            ( model, Cmd.none )

        -- Event coming from Rust (key presses, etc.)
        TerminalEvent (Ok val) ->
            case D.decodeValue (D.field "type" D.string) val of
                Ok "keyPress" ->
                    case D.decodeValue (D.field "key" D.string) val of
                        Ok key ->
                            update (KeyPressed key) model

                        Err _ ->
                            ( model, Cmd.none )

                Ok "exworkResult" ->
                     case D.decodeValue (D.field "payload" exWorkResultDecoder) val of
                        Ok result ->
                            update (ExWorkResult (Ok result)) model
                        Err e ->
                             update (ExWorkResult (Err e)) model
                
                _ ->
                    ( model, Cmd.none )

        TerminalEvent (Err e) ->
            ( { model | status = "JSON Decode Error from runtime: " ++ D.errorToString e }, Cmd.none )

        -- Handle a specific key press
        KeyPressed key ->
            case key of
                "q" ->
                    -- A special command to tell Terminus to exit
                    ( model, executeCommand (E.object [("type", E.string "exit")]) )

                "e" ->
                    update SendEchoCommand model

                _ ->
                    ( { model | status = "Key Pressed: " ++ key }, Cmd.none )
        
        -- A command was successful or failed
        ExWorkResult (Ok result) ->
            ( { model | lastResult = Just result.status_message, status = "IDLE" }, Cmd.none )

        ExWorkResult (Err e) ->
            ( { model | lastResult = Just ("Failed to parse ExWork result: " ++ D.errorToString e), status = "ERROR" }, Cmd.none )
        
        -- Build and send a command to ExWork
        SendEchoCommand ->
            let
                exworkAction =
                    E.object
                        [ ( "type", E.string "ECHO" )
                        , ( "parameters", E.object [ ( "message", E.string "Hello from Elmwood via ExWork!" ) ] )
                        ]

                exworkInstruction =
                    E.object
                        [ ( "step_id", E.string "elm-echo-01" )
                        , ( "actions", E.list identity [ exworkAction ] )
                        ]
                
                commandPayload =
                    E.object 
                        [ ("type", E.string "executeExWork")
                        , ("payload", exworkInstruction)
                        ]
            in
            ( { model | status = "SENDING COMMAND..." }, executeCommand commandPayload )


-- SUBSCRIPTIONS
subscriptions : Model -> Sub Msg
subscriptions _ =
    terminalEvents TerminalEvent


-- VIEW
view : Model -> V.Document Msg
view model =
    V.document [ Attrs.padding 1 ]
        [ V.vstack []
            [ V.box [ Attrs.borderRounded, Attrs.padding 1 ]
                [ V.vstack []
                    [ V.text "--- Elmwood + ExWork Control Interface ---"
                    , V.separator
                    , V.hstack []
                        [ V.text "Status: "
                        , V.text <| model.status
                        ]
                    , V.hstack []
                        [ V.text "Last Result: "
                        , V.text <| Maybe.withDefault "N/A" model.lastResult
                        ]
                    ]
                ]
            , V.box [ Attrs.padding 1 ]
                [ V.text "COMMANDS: (e) Send Echo Command | (q) Quit" ]
   
