#!/usr/bin/env python3
"""
Helper script for running BLE peripheral as external process.

This runs as a separate process to avoid fork/spawn issues with
dynamically loaded tools and Objective-C frameworks.
"""

import sys
import time
import logging
import signal

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(processName)s] - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Characteristic UUIDs
RX_CHAR_UUID = "12345678-1234-5678-1234-56789abcdef1"
TX_CHAR_UUID = "12345678-1234-5678-1234-56789abcdef2"


def run_corebluetooth_peripheral(service_name, service_uuid, system_prompt):
    """Run CoreBluetooth peripheral."""
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
        sys.exit(1)

    # Track state
    running = {"value": False, "stop_requested": False}

    # Delegate
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
            state = peripheral.state()
            logger.info(f"📡 State: {state}")

            if state == 5:  # CBManagerStatePoweredOn
                logger.info("🟢 Powered on! Setting up service...")
                self.setup_service()

        def setup_service(self):
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

            service_uuid_obj = CBUUID.UUIDWithString_(service_uuid)
            self.service = CBMutableService.alloc().initWithType_primary_(
                service_uuid_obj, True
            )
            self.service.setCharacteristics_([self.rx_char, self.tx_char])

            self.peripheral_manager.addService_(self.service)

        def peripheralManager_didAddService_error_(self, peripheral, service, error):
            if error:
                logger.error(f"❌ Error: {error}")
                return

            logger.info("✅ Service added! Starting advertising...")
            adv_data = {
                "kCBAdvDataLocalName": service_name,
                "kCBAdvDataServiceUUIDs": [CBUUID.UUIDWithString_(service_uuid)],
            }
            peripheral.startAdvertising_(adv_data)

        def peripheralManagerDidStartAdvertising_error_(self, peripheral, error):
            if error:
                logger.error(f"❌ Advertising error: {error}")
                return

            logger.info(f"🎉 Advertising '{service_name}'!")
            running["value"] = True
            # Signal parent via stdout
            print("PERIPHERAL_READY", flush=True)

        def peripheralManager_didReceiveWriteRequests_(self, peripheral, requests):
            for request in requests:
                if request.characteristic().UUID().UUIDString() == RX_CHAR_UUID:
                    data = bytes(request.value())
                    logger.info(f"📨 Received: {data}")

                    # Echo response
                    response = f"Echo: {data.decode()}"
                    response_bytes = response.encode()

                    if self.tx_char:
                        self.peripheral_manager.updateValue_forCharacteristic_onSubscribedCentrals_(
                            response_bytes, self.tx_char, None
                        )
                        logger.info(f"📤 Sent: {response[:50]}")

                    peripheral.respondToRequest_withResult_(request, 0)

    # Handle signals
    def signal_handler(sig, frame):
        logger.info("🛑 Stop signal received")
        running["stop_requested"] = True

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    # Create delegate and manager
    delegate = PeripheralDelegate.alloc().init()
    delegate.peripheral_manager = CBPeripheralManager.alloc().initWithDelegate_queue_(
        delegate, None
    )

    # Run loop
    logger.info("Starting NSRunLoop...")
    run_loop = NSRunLoop.currentRunLoop()

    start_time = time.time()
    while time.time() - start_time < 3600 and not running["stop_requested"]:
        run_loop.runMode_beforeDate_(
            NSDefaultRunLoopMode, NSDate.dateWithTimeIntervalSinceNow_(0.1)
        )

    # Cleanup
    logger.info("Cleaning up...")
    if delegate.peripheral_manager:
        delegate.peripheral_manager.stopAdvertising()
    logger.info("Stopped")


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print(
            "Usage: _bluetooth_peripheral_helper.py <service_name> <service_uuid> <system_prompt>"
        )
        sys.exit(1)

    service_name = sys.argv[1]
    service_uuid = sys.argv[2]
    system_prompt = sys.argv[3]

    logger.info(f"Starting peripheral: {service_name}")
    run_corebluetooth_peripheral(service_name, service_uuid, system_prompt)
