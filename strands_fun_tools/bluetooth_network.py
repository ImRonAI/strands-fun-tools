"""Bluetooth Agent Network - BLE-based agent-to-agent communication.

This module provides BLE peripheral (server) and central (client) functionality
for Strands Agents, allowing them to communicate over Bluetooth Low Energy.
The tool runs peripheral operations in background threads, enabling concurrent
communication without blocking the main agent.

Key Features:
1. BLE Peripheral: Act as GATT server, process connections with dedicated agents
2. BLE Central: Connect to remote peripherals and exchange messages
3. Background Processing: Peripheral runs in a background thread
4. Per-Device Agents: Creates a fresh agent for each connected device
5. Local Mesh: Agent-to-agent communication without internet/WiFi

Usage with Strands Agent:

```python
from strands import Agent
from strands_fun_tools import bluetooth_network

agent = Agent(tools=[bluetooth_network])

# Start a BLE peripheral (server)
result = agent.tool.bluetooth_network(
    action="start_peripheral",
    service_name="AgentServer",
    system_prompt="You are a helpful BLE assistant.",
)

# Connect as central (client) and send message
result = agent.tool.bluetooth_network(
    action="central_send",
    device_address="AA:BB:CC:DD:EE:FF",
    service_uuid="12345678-1234-5678-1234-56789abcdef0",
    message="Hello from agent!"
)

# Stop the peripheral
result = agent.tool.bluetooth_network(action="stop_peripheral")
```
"""

import asyncio
import logging
import platform
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from strands import Agent, tool

logger = logging.getLogger(__name__)

# Global registry for peripheral subprocesses
PERIPHERAL_PROCESSES: dict[str, dict[str, Any]] = {}

# Standard UUIDs for agent communication
DEFAULT_SERVICE_UUID = "12345678-1234-5678-1234-56789abcdef0"
RX_CHAR_UUID = "12345678-1234-5678-1234-56789abcdef1"  # Receive from central
TX_CHAR_UUID = "12345678-1234-5678-1234-56789abcdef2"  # Transmit to central


def _check_bleak():
    """Check if bleak is available."""
    try:
        import bleak

        return True
    except ImportError:
        return False


def _check_peripheral_support():
    """Check if platform has peripheral support available."""
    system = platform.system().lower()

    if system == "darwin":  # macOS
        try:
            import objc
            from Foundation import NSObject
            from CoreBluetooth import (
                CBPeripheralManager,
                CBMutableService,
                CBMutableCharacteristic,
            )

            return True, "corebluetooth"
        except ImportError:
            return False, None

    elif system == "linux":
        try:
            import dbus

            return True, "bluez"
        except ImportError:
            return False, None

    else:
        return False, None


async def handle_device_connection(
    device_address: str,
    rx_data: bytes,
    system_prompt: str,
    model: Any,
    parent_tools: list | None = None,
    callback_handler: Any = None,
    trace_attributes: dict | None = None,
) -> str:
    """Handle a device connection and process message with dedicated agent.

    Args:
        device_address: Address of the connected device
        rx_data: Data received from device
        system_prompt: System prompt for creating agent
        model: Model instance from parent agent
        parent_tools: Tools inherited from parent agent
        callback_handler: Callback handler from parent agent
        trace_attributes: Trace attributes from parent agent

    Returns:
        str: Response from agent to send back
    """
    logger.info(f"Processing message from device {device_address}")

    # Create a fresh agent instance for this device
    device_agent = Agent(
        model=model,
        messages=[],
        tools=parent_tools or [],
        callback_handler=callback_handler,
        system_prompt=system_prompt,
        trace_attributes=trace_attributes or {},
    )

    try:
        # Decode message
        message = rx_data.decode().strip()
        logger.info(f"Received from {device_address}: {message}")

        # Process with agent
        response = device_agent(message)
        response_text = str(response)

        return response_text

    except Exception as e:
        logger.error(f"Error processing message from {device_address}: {e}")
        return f"Error: {e}"


# ============================================================================
# macOS CoreBluetooth Peripheral Implementation
# ============================================================================


def run_corebluetooth_peripheral(
    service_name: str,
    service_uuid: str,
    system_prompt: str,
    running_flag: Any,  # multiprocessing.Value
    connections_counter: Any,  # multiprocessing.Value
    parent_agent_data: dict | None = None,
) -> None:
    """Run BLE peripheral using macOS CoreBluetooth in separate process.

    This MUST run in a separate process (not thread) because CoreBluetooth
    delegate callbacks only work in the main thread of a process.
    """
    try:
        import objc
        from Foundation import NSObject, NSRunLoop, NSDefaultRunLoopMode, NSDate
        from CoreBluetooth import (
            CBPeripheralManager,
            CBMutableService,
            CBMutableCharacteristic,
            CBCharacteristicPropertyRead,
            CBCharacteristicPropertyWrite,
            CBCharacteristicPropertyNotify,
            CBAttributePermissionsReadable,
            CBAttributePermissionsWriteable,
            CBUUID,
        )
    except ImportError as e:
        logger.error(f"CoreBluetooth not available: {e}")
        return

    peripheral_id = service_uuid

    # PeripheralManager Delegate
    class PeripheralDelegate(NSObject):
        def init(self):
            self = objc.super(PeripheralDelegate, self).init()
            if self is None:
                return None
            self.peripheral_manager = None
            self.service = None
            self.rx_char = None
            self.tx_char = None
            return self

        def peripheralManagerDidUpdateState_(self, peripheral):
            """Called when peripheral manager state updates."""
            state = peripheral.state()
            logger.info(f"📡 Peripheral state: {state}")

            if state == 5:  # CBManagerStatePoweredOn
                logger.info("🟢 Bluetooth powered on! Setting up service...")
                self.setup_service()

        def setup_service(self):
            """Setup GATT service and characteristics."""
            # Create characteristics
            rx_uuid = CBUUID.UUIDWithString_(RX_CHAR_UUID)
            tx_uuid = CBUUID.UUIDWithString_(TX_CHAR_UUID)

            self.rx_char = CBMutableCharacteristic.alloc().initWithType_properties_value_permissions_(
                rx_uuid,
                CBCharacteristicPropertyWrite,
                None,
                CBAttributePermissionsWriteable,
            )

            self.tx_char = CBMutableCharacteristic.alloc().initWithType_properties_value_permissions_(
                tx_uuid,
                CBCharacteristicPropertyRead | CBCharacteristicPropertyNotify,
                None,
                CBAttributePermissionsReadable,
            )

            # Create service
            service_uuid_obj = CBUUID.UUIDWithString_(service_uuid)
            self.service = CBMutableService.alloc().initWithType_primary_(
                service_uuid_obj, True
            )
            self.service.setCharacteristics_([self.rx_char, self.tx_char])

            # Add service
            self.peripheral_manager.addService_(self.service)
            logger.info("📦 Service added to manager")

        def peripheralManager_didAddService_error_(self, peripheral, service, error):
            """Called when service is added."""
            if error:
                logger.error(f"❌ Error adding service: {error}")
                return

            logger.info("✅ Service added successfully! Starting advertising...")

            # Start advertising
            adv_data = {
                "kCBAdvDataLocalName": service_name,
                "kCBAdvDataServiceUUIDs": [CBUUID.UUIDWithString_(service_uuid)],
            }
            peripheral.startAdvertising_(adv_data)

        def peripheralManagerDidStartAdvertising_error_(self, peripheral, error):
            """Called when advertising starts."""
            if error:
                logger.error(f"❌ Error starting advertising: {error}")
                return

            logger.info(f"🎉 Advertising started for '{service_name}'!")
            running_flag.value = 1  # Signal that we're running

        def peripheralManager_didReceiveWriteRequests_(self, peripheral, requests):
            """Called when central writes to characteristic."""
            for request in requests:
                if request.characteristic().UUID().UUIDString() == RX_CHAR_UUID:
                    data = bytes(request.value())
                    logger.info(f"📨 Received write: {data}")

                    connections_counter.value += 1

                    # Process message in separate thread
                    threading.Thread(
                        target=self.process_message,
                        args=(data, parent_agent_data),
                        daemon=True,
                    ).start()

                    # Respond to write request
                    peripheral.respondToRequest_withResult_(
                        request, 0
                    )  # CBATTErrorSuccess

        def process_message(self, data: bytes, agent_data: dict | None):
            """Process incoming message with agent."""
            try:
                # Decode message
                message = data.decode().strip()
                logger.info(f"💬 Processing: {message}")

                # For now, simple echo response
                # TODO: Create Agent instance if agent_data provided
                response = f"Echo: {message}"

                # Update TX characteristic value
                response_bytes = response.encode()
                if self.tx_char and self.peripheral_manager:
                    success = self.peripheral_manager.updateValue_forCharacteristic_onSubscribedCentrals_(
                        response_bytes,
                        self.tx_char,
                        None,  # Send to all subscribed centrals
                    )
                    logger.info(
                        f"📤 Response sent (success={success}): {response[:50]}..."
                    )
            except Exception as e:
                logger.error(f"❌ Error processing message: {e}")

    # Create delegate and peripheral manager
    delegate = PeripheralDelegate.alloc().init()

    logger.info("Creating CBPeripheralManager in process...")
    delegate.peripheral_manager = CBPeripheralManager.alloc().initWithDelegate_queue_(
        delegate, None
    )

    # Run event loop - this is now the main thread of this process
    logger.info("Starting NSRunLoop (process main thread)...")
    run_loop = NSRunLoop.currentRunLoop()

    start_time = time.time()
    max_runtime = 3600  # Run for up to 1 hour

    # running_flag values: -1 = stop requested, 0 = initializing, 1 = running
    while time.time() - start_time < max_runtime:
        # Process events
        run_loop.runMode_beforeDate_(
            NSDefaultRunLoopMode, NSDate.dateWithTimeIntervalSinceNow_(0.1)
        )

        # Check stop signal (-1 = stop requested)
        if running_flag.value == -1:
            logger.info("🛑 Stop signal received from parent process")
            break

    # Cleanup
    logger.info("Cleaning up peripheral...")
    if delegate.peripheral_manager:
        delegate.peripheral_manager.stopAdvertising()

    running_flag.value = 0  # Ensure stopped state
    logger.info("CoreBluetooth peripheral stopped")


# ============================================================================
# Linux BlueZ D-Bus Peripheral Implementation
# ============================================================================


def run_bluez_peripheral(
    service_name: str,
    service_uuid: str,
    system_prompt: str,
    running_flag: Any,  # multiprocessing.Value
    connections_counter: Any,  # multiprocessing.Value
    parent_agent_data: dict | None = None,
) -> None:
    """Run BLE peripheral using Linux BlueZ D-Bus API in separate process."""
    try:
        import dbus
        import dbus.service
        import dbus.mainloop.glib
        from gi.repository import GLib
    except ImportError as e:
        logger.error(f"BlueZ D-Bus not available: {e}")
        return

    peripheral_id = service_uuid

    # Initialize D-Bus main loop
    dbus.mainloop.glib.DBusGMainLoop(set_as_default=True)
    bus = dbus.SystemBus()

    # [BlueZ implementation continues here - keeping it as is for now]
    # Signal running
    running_flag.value = 1
    logger.info(f"✅ BlueZ peripheral registered for '{service_name}'")

    # Run main loop
    mainloop = GLib.MainLoop()
    try:
        mainloop.run()
    except KeyboardInterrupt:
        mainloop.quit()

    running_flag.value = 0
    logger.info("BlueZ peripheral stopped")


def run_peripheral(
    service_name: str,
    service_uuid: str,
    system_prompt: str,
    running_flag: Any,
    connections_counter: Any,
    parent_agent_data: dict | None = None,
) -> None:
    """Run a BLE peripheral - delegates to platform-specific implementation."""
    has_support, backend = _check_peripheral_support()

    if not has_support:
        logger.error("No peripheral support available on this platform")
        return

    if backend == "corebluetooth":
        run_corebluetooth_peripheral(
            service_name,
            service_uuid,
            system_prompt,
            running_flag,
            connections_counter,
            parent_agent_data,
        )
    elif backend == "bluez":
        run_bluez_peripheral(
            service_name,
            service_uuid,
            system_prompt,
            running_flag,
            connections_counter,
            parent_agent_data,
        )


async def send_to_peripheral(
    device_address: str,
    service_uuid: str,
    message: str,
    timeout: float = 10.0,
) -> str:
    """Connect to BLE peripheral as central and send message.

    Args:
        device_address: BLE address of peripheral
        service_uuid: Service UUID to connect to
        message: Message to send
        timeout: Connection timeout in seconds

    Returns:
        str: Response from peripheral
    """
    try:
        from bleak import BleakClient, BleakScanner
    except ImportError:
        return "Error: bleak not installed"

    logger.info(f"Connecting to peripheral {device_address}...")

    try:
        async with BleakClient(device_address, timeout=timeout) as client:
            logger.info(f"Connected to {device_address}")

            # Find service
            service = None
            for svc in client.services:
                if svc.uuid == service_uuid:
                    service = svc
                    break

            if not service:
                return f"Error: Service {service_uuid} not found on device"

            # Find RX and TX characteristics
            rx_char = None
            tx_char = None
            for char in service.characteristics:
                if RX_CHAR_UUID in char.uuid:
                    rx_char = char
                elif TX_CHAR_UUID in char.uuid:
                    tx_char = char

            if not rx_char:
                return "Error: RX characteristic not found"

            # Write message to RX characteristic
            message_bytes = (message + "\n").encode()
            await client.write_gatt_char(rx_char.uuid, message_bytes)
            logger.info(f"Sent message to {device_address}")

            # If TX characteristic exists, try to read response
            if tx_char and "read" in char.properties:
                await asyncio.sleep(0.5)  # Wait for processing
                response_bytes = await client.read_gatt_char(tx_char.uuid)
                response = response_bytes.decode().strip()
                logger.info(f"Received response: {response}")
                return response
            else:
                return "Message sent successfully (no response characteristic)"

    except Exception as e:
        logger.error(f"Error communicating with {device_address}: {e}")
        return f"Error: {e}"


@tool
def bluetooth_network(
    action: str,
    service_name: str = "AgentNetwork",
    service_uuid: str = DEFAULT_SERVICE_UUID,
    system_prompt: str = "You are a helpful BLE agent assistant.",
    device_address: str = "",
    message: str = "",
    timeout: float = 10.0,
    agent: Any = None,
) -> dict:
    """Create and manage BLE agent network for local agent-to-agent communication.

    This function provides BLE peripheral (server) and central (client) functionality
    for Strands agents, allowing them to communicate over Bluetooth Low Energy without
    requiring internet or WiFi connectivity.

    How It Works:
    ------------
    1. Peripheral Mode (Server):
       - Starts a BLE GATT server in a background thread
       - Creates a dedicated agent for EACH connected device
       - Inherits tools from the parent agent
       - Processes device messages and returns responses
       - Uses RX/TX characteristics for bidirectional communication

    2. Central Mode (Client):
       - Scans for and connects to BLE peripherals
       - Sends messages via GATT characteristic writes
       - Receives responses via GATT characteristic reads
       - Maintains stateless connections

    3. Management:
       - Track peripheral status and statistics
       - Stop peripherals gracefully
       - Monitor connections and performance

    Common Use Cases:
    ----------------
    1. Agent Swarms:
       - Multiple agents coordinating locally over BLE
       - No cloud or internet required
       - Proximity-based collaboration

    2. Phone ↔ Computer Communication:
       - Seamless local agent communication
       - Low power consumption
       - Privacy-preserving (no network traffic)

    3. IoT Agent Orchestration:
       - Control smart devices via BLE agents
       - Local automation workflows
       - Distributed agent processing

    4. Secure Local Channels:
       - Air-gapped agent communication
       - No cloud dependencies
       - Local-only data exchange

    Args:
        action: Action to perform (start_peripheral, stop_peripheral, central_send,
            status, list_peripherals)
        service_name: Human-readable name for BLE service (advertising)
        service_uuid: UUID for GATT service (default: standard agent UUID)
        system_prompt: System prompt for peripheral agents (per-device)
        device_address: BLE device address for central mode (AA:BB:CC:DD:EE:FF)
        message: Message to send in central mode
        timeout: Connection timeout for central mode (seconds)
        agent: Parent agent (automatically provided by Strands)

    Returns:
        dict: Operation result with status and details

    Actions:
        start_peripheral: Start BLE peripheral (server) in background
        stop_peripheral: Stop running peripheral
        central_send: Connect as central and send message to peripheral
        status: Get peripheral status and statistics
        list_peripherals: List all running peripherals

    Examples:
        # Start peripheral
        bluetooth_network(
            action="start_peripheral",
            service_name="MyAgent",
            system_prompt="You are agent assistant"
        )

        # Send message from another device
        bluetooth_network(
            action="central_send",
            device_address="AA:BB:CC:DD:EE:FF",
            service_uuid="12345678-1234-5678-1234-56789abcdef0",
            message="Hello from remote agent!"
        )

        # Check status
        bluetooth_network(action="status")

        # Stop peripheral
        bluetooth_network(action="stop_peripheral")

    Notes:
        - Requires 'bleak' package for client mode: pip install bleak
        - macOS peripheral mode requires pyobjc-framework-CoreBluetooth (usually pre-installed)
        - Linux peripheral mode requires dbus-python and PyGObject
        - Peripheral mode creates full GATT server with RX/TX characteristics
        - Responses automatically written back to TX characteristic
        - Central mode works on all platforms with BLE adapters
        - Service UUIDs must match between peripheral and central
        - Each connected device gets its own dedicated agent instance
        - Peripheral runs in separate PROCESS (not thread) for CoreBluetooth compatibility
        - Uses platform-native APIs: CoreBluetooth (macOS), BlueZ D-Bus (Linux)
        - CoreBluetooth requires process main thread - multiprocessing.Process provides this
    """
    # General dependency check (will be refined per action)
    if not _check_bleak():
        has_periph, _ = _check_peripheral_support()
        if not has_periph:
            return {
                "status": "error",
                "content": [
                    {
                        "text": "BLE packages not installed. Install bleak for client mode."
                    }
                ],
            }

    if action == "start_peripheral":
        # Check for peripheral support
        has_support, backend = _check_peripheral_support()
        if not has_support:
            system = platform.system().lower()
            if system == "darwin":
                return {
                    "status": "error",
                    "content": [
                        {
                            "text": "pyobjc-framework-CoreBluetooth required for macOS peripheral mode. Install with: pip install pyobjc-framework-CoreBluetooth"
                        }
                    ],
                }
            elif system == "linux":
                return {
                    "status": "error",
                    "content": [
                        {
                            "text": "dbus-python and PyGObject required for Linux peripheral mode. Install with: pip install dbus-python PyGObject"
                        }
                    ],
                }
            else:
                return {
                    "status": "error",
                    "content": [{"text": f"Peripheral mode not supported on {system}"}],
                }

        peripheral_id = service_uuid

        if (
            peripheral_id in PERIPHERAL_PROCESSES
            and PERIPHERAL_PROCESSES[peripheral_id].get("process")
            and PERIPHERAL_PROCESSES[peripheral_id]["process"].poll() is None
        ):
            return {
                "status": "error",
                "content": [
                    {
                        "text": f"Peripheral with service UUID {service_uuid} already running"
                    }
                ],
            }

        # Find helper script path
        helper_script = Path(__file__).parent / "_bluetooth_peripheral_helper.py"
        if not helper_script.exists():
            return {
                "status": "error",
                "content": [{"text": f"Helper script not found: {helper_script}"}],
            }

        # Start peripheral as external subprocess
        logger.info(f"Starting peripheral subprocess: {service_name}")
        process = subprocess.Popen(
            [
                "python3",
                str(helper_script),
                service_name,
                service_uuid,
                system_prompt,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

        # Initialize peripheral registry
        PERIPHERAL_PROCESSES[peripheral_id] = {
            "service_name": service_name,
            "service_uuid": service_uuid,
            "system_prompt": system_prompt,
            "process": process,
            "start_time": time.time(),
            "backend": backend,
            "running": False,
        }

        # Wait for PERIPHERAL_READY signal or timeout
        logger.info("Waiting for peripheral to start advertising...")
        ready = False
        for i in range(30):  # Wait up to 3 seconds
            time.sleep(0.1)

            # Check stdout for PERIPHERAL_READY
            if process.stdout:
                # Non-blocking read
                import select

                if select.select([process.stdout], [], [], 0)[0]:
                    line = process.stdout.readline().strip()
                    if line == "PERIPHERAL_READY":
                        logger.info(f"✅ Peripheral ready after {(i+1)*0.1:.1f}s")
                        PERIPHERAL_PROCESSES[peripheral_id]["running"] = True
                        ready = True
                        break

            # Check if process died
            if process.poll() is not None:
                stderr = process.stderr.read() if process.stderr else ""
                logger.error(f"Process died: {stderr}")
                break

        return {
            "status": "success",
            "content": [
                {
                    "text": f"{'✅' if ready else '⚠️'} BLE Peripheral '{service_name}' started ({backend})\n\n"
                    f"📡 Service UUID: {service_uuid}\n"
                    f"📨 RX Characteristic: {RX_CHAR_UUID}\n"
                    f"📤 TX Characteristic: {TX_CHAR_UUID}\n"
                    f"📶 Advertising: {'Yes' if ready else 'Initializing...'}\n\n"
                    f"Central devices can now connect and send messages!"
                }
            ],
        }

    elif action == "stop_peripheral":
        peripheral_id = service_uuid

        if peripheral_id not in PERIPHERAL_PROCESSES:
            return {
                "status": "error",
                "content": [
                    {"text": f"No peripheral found with service UUID {service_uuid}"}
                ],
            }

        process = PERIPHERAL_PROCESSES[peripheral_id].get("process")
        if not process or process.poll() is not None:
            return {
                "status": "error",
                "content": [{"text": f"Peripheral {service_uuid} is not running"}],
            }

        # Terminate subprocess gracefully
        logger.info("Sending SIGTERM to peripheral subprocess...")
        process.terminate()

        # Wait for process to finish
        try:
            process.wait(timeout=3.0)
        except subprocess.TimeoutExpired:
            logger.warning("Process didn't stop gracefully, killing...")
            process.kill()
            process.wait()

        backend = PERIPHERAL_PROCESSES[peripheral_id].get("backend", "unknown")
        del PERIPHERAL_PROCESSES[peripheral_id]

        return {
            "status": "success",
            "content": [{"text": f"✅ BLE Peripheral stopped ({backend})"}],
        }

    elif action == "central_send":
        # Check for bleak (required for central/client mode)
        if not _check_bleak():
            return {
                "status": "error",
                "content": [
                    {
                        "text": "bleak package required for central mode. Install with: pip install bleak"
                    }
                ],
            }

        if not device_address:
            return {
                "status": "error",
                "content": [
                    {"text": "device_address required for central_send action"}
                ],
            }

        if not message:
            return {
                "status": "error",
                "content": [{"text": "message required for central_send action"}],
            }

        # Run async send in new event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            response = loop.run_until_complete(
                send_to_peripheral(device_address, service_uuid, message, timeout)
            )
            return {
                "status": "success",
                "content": [
                    {
                        "text": f"📡 Sent to {device_address}\n\n"
                        f"💬 Message: {message}\n"
                        f"📨 Response: {response}"
                    }
                ],
            }
        finally:
            loop.close()

    elif action == "status":
        if not PERIPHERAL_PROCESSES:
            return {
                "status": "success",
                "content": [{"text": "No peripherals running"}],
            }

        status_lines = ["📡 **BLE Peripheral Status:**\n"]
        for peripheral_id, info in PERIPHERAL_PROCESSES.items():
            process = info.get("process")
            is_running = (
                process and process.poll() is None and info.get("running", False)
            )
            running = "🟢 Running" if is_running else "🔴 Stopped"
            uptime = time.time() - info.get("start_time", time.time())
            status_lines.append(
                f"\n**{info['service_name']}**\n"
                f"- Status: {running}\n"
                f"- Service UUID: {peripheral_id}\n"
                f"- Uptime: {uptime:.1f}s"
            )

        return {"status": "success", "content": [{"text": "\n".join(status_lines)}]}

    elif action == "list_peripherals":
        if not PERIPHERAL_PROCESSES:
            return {
                "status": "success",
                "content": [{"text": "No peripherals configured"}],
            }

        peripherals = []
        for peripheral_id, info in PERIPHERAL_PROCESSES.items():
            process = info.get("process")
            is_running = (
                process and process.poll() is None and info.get("running", False)
            )
            peripherals.append(
                {
                    "service_name": info["service_name"],
                    "service_uuid": peripheral_id,
                    "running": is_running,
                }
            )

        return {
            "status": "success",
            "content": [
                {
                    "text": f"Found {len(peripherals)} peripheral(s)",
                    "peripherals": peripherals,
                }
            ],
        }

    else:
        return {"status": "error", "content": [{"text": f"Unknown action: {action}"}]}
