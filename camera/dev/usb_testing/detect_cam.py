import usb.core as uc
import usb.util as uu
import subprocess

class find_class(object):
    def __init__(self, class_):
        self._class = class_
    def __call__(self, device):
        if device.bDeviceClass == self._class:
            return True
        for cfg in device:
            intf = uu.find_descriptor(cfg, bInterfaceClass=self._class)
            if intf is not None:
                return True
        return False

# Find all USB devices with video interface class (14)
devices = uc.find(find_all=True, custom_match=find_class(14))

# Get the v4l control
v4ctl = subprocess.check_output("v4l2-ctl --list-devices", shell=True).decode("utf-8")
v4ctl = v4ctl.split("\n")

matches = []

# Print the devices
for device in devices:
    print(device.bus, device.address, device.port_number)
    try:
        output = subprocess.check_output(f'lsusb -tvv | grep /dev/bus/usb/{device.bus:03}/{device.address:03}', shell=True).decode("utf-8")
        id = output.split("\n")[0].strip().split(" ")[0].split("1-", 1)[-1]
        for i in range(len(v4ctl)):
            if id in v4ctl[i]:
                matches.append((f'Device at bus and address: {device.bus}, {device.address}',v4ctl[i+1].strip()))
                break
    except:
        pass

print(matches)