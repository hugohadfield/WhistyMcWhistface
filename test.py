import win32gui
import win32ui
from ctypes import windll
from PIL import Image

def get_hwnd_from_string(hwnd_string):
	hwnd = win32gui.FindWindowEx(None, 0, None, hwnd_string)
	return hwnd

def screengrab(hwnd):
	left, top, right, bottom = win32gui.GetWindowRect(hwnd)
	width, height = right - left, bottom - top
	hwndDC = win32gui.GetWindowDC(hwnd)
	mfcDC  = win32ui.CreateDCFromHandle(hwndDC)
	saveDC = mfcDC.CreateCompatibleDC()
	saveDC.SetWindowOrg((13, 151))
	saveBitMap = win32ui.CreateBitmap()

	saveBitMap.CreateCompatibleBitmap(mfcDC, 287, 76)
	saveDC.SelectObject(saveBitMap)
	result = windll.user32.PrintWindow(hwnd, saveDC.GetSafeHdc(), 0)

	bmpinfo = saveBitMap.GetInfo()
	bmpstr = saveBitMap.GetBitmapBits(True)
	im = Image.frombuffer(
	    'RGB',
	    (bmpinfo['bmWidth'], bmpinfo['bmHeight']),
	    bmpstr, 'raw', 'BGRX', 0, 1)

	win32gui.DeleteObject(saveBitMap.GetHandle())
	saveDC.DeleteDC()
	mfcDC.DeleteDC()
	win32gui.ReleaseDC(hwnd, hwndDC)
	if result != 0:
		im.save("image.bmp")
	else:
		# We couldn't get an image :( - handle it as you wish.
		print " fail "

hwnd = get_hwnd_from_string("QWOP - Google Chrome")
screengrab(hwnd)