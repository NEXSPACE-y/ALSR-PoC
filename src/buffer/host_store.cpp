#include "alsr/host_store.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#include <sstream>

namespace alsr {

std::string_view to_string(HostStoreMode m) noexcept {
  switch (m) {
    case HostStoreMode::Mmap:      return "mmap";
    case HostStoreMode::Read:      return "read";
    case HostStoreMode::Synthetic: return "synthetic";
  }
  return "unknown";
}

HostStoreMode host_store_mode_from_string(std::string_view s) {
  if (s == "mmap")      return HostStoreMode::Mmap;
  if (s == "read")      return HostStoreMode::Read;
  if (s == "synthetic") return HostStoreMode::Synthetic;
  fail(ErrorKind::Invalid,
       "unknown host store mode '" + std::string(s) + "' (mmap|read|synthetic)");
}

HostStore::HostStore(std::string path, HostStoreMode mode, Bytes synthetic_size)
    : path_(std::move(path)), mode_(mode) {
  if (mode_ == HostStoreMode::Synthetic) {
    file_size_ = synthetic_size;
    return;
  }

  // NFR-03: 読み取り専用でのみ開く。
  fd_ = ::open(path_.c_str(), O_RDONLY);
  if (fd_ < 0) {
    fail(ErrorKind::Io, "open(O_RDONLY) failed for " + path_ + ": " +
                            std::strerror(errno));
  }
  struct stat st {};
  if (::fstat(fd_, &st) != 0) {
    ::close(fd_);
    fd_ = -1;
    fail(ErrorKind::Io, "fstat failed for " + path_);
  }
  file_size_ = static_cast<Bytes>(st.st_size);

  if (mode_ == HostStoreMode::Mmap) {
    if (file_size_ > 0) {
      map_len_ = static_cast<std::size_t>(file_size_);
      map_ = ::mmap(nullptr, map_len_, PROT_READ, MAP_PRIVATE, fd_, 0);
      if (map_ == MAP_FAILED) {
        map_ = nullptr;
        ::close(fd_);
        fd_ = -1;
        fail(ErrorKind::Io,
             "mmap failed for " + path_ + ": " + std::strerror(errno));
      }
    }
  }
}

HostStore::~HostStore() {
  if (map_ != nullptr) ::munmap(map_, map_len_);
  if (fd_ >= 0) ::close(fd_);
}

void HostStore::read_into(Bytes offset, Bytes len, u8* dst) {
  if (len == 0) return;
  if (dst == nullptr) fail(ErrorKind::Invalid, "read_into: dst is null");
  if (offset > file_size_ || len > file_size_ - offset) {
    std::ostringstream os;
    os << "read_into out of range: offset=" << offset << " len=" << len
       << " file_size=" << file_size_;
    fail(ErrorKind::Invalid, os.str());
  }

  switch (mode_) {
    case HostStoreMode::Synthetic: {
      for (Bytes k = 0; k < len; ++k) dst[k] = synthetic_byte(offset + k);
      break;
    }
    case HostStoreMode::Mmap: {
      if (map_ == nullptr) fail(ErrorKind::State, "mmap region is not available");
      std::memcpy(dst, static_cast<const u8*>(map_) + offset,
                  static_cast<std::size_t>(len));
      break;
    }
    case HostStoreMode::Read: {
      Bytes done = 0;
      while (done < len) {
        const ssize_t got =
            ::pread(fd_, dst + done, static_cast<std::size_t>(len - done),
                    static_cast<off_t>(offset + done));
        if (got < 0) {
          if (errno == EINTR) continue;
          fail(ErrorKind::Io, "pread failed on " + path_ + ": " +
                                  std::strerror(errno));
        }
        if (got == 0) {
          std::ostringstream os;
          os << "unexpected EOF at offset " << (offset + done);
          fail(ErrorKind::Io, os.str());
        }
        done += static_cast<Bytes>(got);
      }
      break;
    }
  }
  bytes_read_ += len;
}

}  // namespace alsr
