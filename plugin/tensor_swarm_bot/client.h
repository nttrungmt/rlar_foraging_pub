#pragma once

#include <arpa/inet.h>
#include <chrono>
#include <cstring>
#include <iostream>
#include <string>
#include <sys/socket.h>
#include <unistd.h>
#include <vector>

class Client {
public:
  Client();
  Client(const std::string ip, int port);
  ~Client();

  void Init(const std::string ip = "127.0.0.1", int port = 5001);
  bool Send(const std::string& message) const;
  bool Receive(std::string& message);

private:
  int client_;
  const int size_message_length_ = 8;  // Buffer size for the length
  
  int Send_(const std::string& strData) const;
  int Receive_(char* pData, const size_t uSize, bool bReadFully = true) const;
};
