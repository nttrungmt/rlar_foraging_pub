#include "client.h"
#include <unistd.h>

Client::Client() {}
Client::Client(const std::string ip, int port) { Init(ip, port); }
Client::~Client() { close(client_); }

void Client::Init(const std::string ip, int port) {
    client_ = socket(AF_INET, SOCK_STREAM, 0);
    if (client_ < 0) {
        std::cout << "[Client]: ERROR establishing socket\n" << std::endl;
        exit(-1);
    }

    bool connected = false;
    int connection_attempts = 5;

    while ((!connected) && (connection_attempts > 0)) {
        struct sockaddr_in serv_addr;
        serv_addr.sin_family = AF_INET;
        serv_addr.sin_port = htons(port);
        int s = inet_pton(AF_INET, ip.c_str(), &serv_addr.sin_addr);
        if (s <= 0) {
            if(s==0) {
                printf("Host address [%s] not in correct format\n", ip.c_str());
            } else {
                printf("Invalid address / Address not support \n");
                perror("inet_pton");
            }
            exit(EXIT_FAILURE);
        }

        if (connect(client_, (const struct sockaddr*)&serv_addr, sizeof(serv_addr)) >= 0) {
            connected = true;
            std::cout << "[Client]: Cpp socket client connected." << std::endl;
        } else {
            port += 1;
            connection_attempts -= 1;
            std::cout << "[Client]: Error connecting to port " << port - 1
                << ". Attepting to connect to port: " << port << std::endl;
        }
    }
}

int Client::Send_(const std::string& strData) const
{
    const char* pData = strData.c_str();
    size_t uSize = strData.length();
    //std::cout << "[Client::Send] ready to send " << uSize << " byte" << std::endl;
    int total = 0;
    const int flags = 0;
    int nSent;
    do
    {
        nSent = send(client_, pData + total, uSize - total, flags);
        if (nSent < 0)
        {
            std::cout << "[TCPClient][Error] Socket error in call to send." << std::endl;
            return false;
        }
        total += nSent;
    } while(total < uSize);
    
    return total;
}

bool Client::Send(const std::string& message) const {
    // Send length of the message
    int length = message.length();
    std::string length_str = std::to_string(length);
    std::string message_length =
        std::string(size_message_length_ - length_str.length(), '0') + length_str;
    //send(client_, message_length.c_str(), size_message_length_, 0);
    int bSuccess = Send_(message_length);

    // Send message
    //send(client_, message.c_str(), length, 0);
    if(bSuccess)
        bSuccess = Send_(message);
    return bSuccess>=length;
}

int Client::Receive_(char* pData, const size_t uSize, bool bReadFully) const
{
    if (!pData || !uSize)
        return false;

    int total = 0;
    do
    {
        int nRecvd = recv(client_, pData + total, uSize - total, 0);
        if (nRecvd == 0)
        {
            // peer shut down
            break;
        }
        total += nRecvd;
    } while (bReadFully && (total < uSize));

    return total;
}

bool Client::Receive(std::string& message) {
    // TODO(oleguer): try catch, if connection dropped print notification and try
    // to reconnect
    // Receive length of the message
    char message_length[size_message_length_] = {0};
    //int n = recv(client_, message_length, size_message_length_, 0);
    int n = Receive_(message_length, size_message_length_);
    if (n < size_message_length_)
        return false;
    //get next message length
    std::string message_length_string(message_length);
    int length = std::stoi(message_length_string);
    // if (length == 0) return "";

    // receive next message
    char message_buffer[length] = {0};
    //n = recv(client_, message, length, 0);
    n = Receive_(message_buffer, length);
    if (n < length)
        return false;
    message = message_buffer;
    return true;
}

#ifdef USE_OPENCV
void Client::SendImage(cv::Mat img) {
  int pixel_number = img.rows * img.cols / 2;

  std::vector<uchar> buf(pixel_number);
  cv::imencode(".jpg", img, buf);

  int length = buf.size();
  std::string length_str = std::to_string(length);
  std::string message_length =
      std::string(size_message_length_ - length_str.length(), '0') + length_str;

  send(client_, message_length.c_str(), size_message_length_, 0);
  send(client_, buf.data(), length, 0);
}
#endif
