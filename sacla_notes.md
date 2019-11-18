# SACLA Notes

## Get online access
Easiest way is to connect unsing a ssh socks tunnel (ssh -R 12345) and use proxychains (https://github.com/haad/proxychains)
You need to build proxychains on some linux system or use my buid (http://felixz.de/proxychains_sacla.tar.gz) and copy both the .so as well as the executable to xhpcfep
Save the following config in a file
```
# proxychains.conf
strict_chain
tcp_read_time_out 15000
tcp_connect_time_out 8000
[ProxyList]
socks4 127.0.0.1 12345 
```
and use proxychains like this ./proxychains4 -f _pathToConfigFile_ _commandThatNeedsOnlineAccess_


## Copy files from sacla
The transfer speed is limit by the delay (https://en.wikipedia.org/wiki/Bandwidth-delay_product).
One way around it is to use multiple http connections to download files from SACLA:
- start a http server that supports range requests at SACLA, for example https://github.com/danvk/RangeHTTPServer or the node http server (https://www.npmjs.com/package/http-server) 
- while connected to the VPN use a http download tool that supports multiple connects, eg. aria2 (https://aria2.github.io/)


