#include "cnninst.h"

#ifdef __cplusplus
extern "C" {
#endif

inline void LoadV_Width(uint64_t addr) {
  register uint64_t addr_reg asm ("a0") = addr;
  __asm__ volatile(
    ".word 0x0005000b"
    : :[addr_reg]"r"(addr_reg)
  );
}

inline void LoadV_D_Main(uint64_t addr, uint8_t k, uint8_t vec_addr, uint8_t vtag) {
  register uint64_t addr_reg asm ("a0") = addr;
  switch (k) {
    case 1: {
      __asm__ volatile(
        ".word 0x0015200b"
        : :[addr_reg]"r"(addr_reg)
      );
      break;
    }
    case 2: {
      switch (vec_addr) {
        case 0: {
          __asm__ volatile(
          ".word 0x0025200b"
          : :[addr_reg]"r"(addr_reg)
          );
          break;
        }
        default: {  // case 1
          __asm__ volatile(
          ".word 0x0025208b"
          : :[addr_reg]"r"(addr_reg)
          );
          break;
        }
      }
      break;
    }
    case 3: {
      switch (vec_addr) {
        case 0: {
          __asm__ volatile(
          ".word 0x0035200b"
          : :[addr_reg]"r"(addr_reg)
          );
          break;
        }
        case 1: {
          __asm__ volatile(
          ".word 0x0035208b"
          : :[addr_reg]"r"(addr_reg)
          );
          break;
        }
        default: {  // case 2
          __asm__ volatile(
          ".word 0x0035210b"
          : :[addr_reg]"r"(addr_reg)
          );
          break;
        }
      }
      break;
    }
    case 4: {
      switch (vec_addr) {
        case 0: {
          __asm__ volatile(
          ".word 0x0045200b"
          : :[addr_reg]"r"(addr_reg)
          );
          break;
        }
        case 1: {
          __asm__ volatile(
          ".word 0x0045208b"
          : :[addr_reg]"r"(addr_reg)
          );
          break;
        }
        case 2: {
          __asm__ volatile(
          ".word 0x0045210b"
          : :[addr_reg]"r"(addr_reg)
          );
          break;
        }
        default: {  // case 3
          __asm__ volatile(
          ".word 0x0045218b"
          : :[addr_reg]"r"(addr_reg)
          );
          break;
        }
      }
      break;
    }
    case 5: {
      switch (vec_addr) {
        case 0: {
          __asm__ volatile(
          ".word 0x0055200b"
          : :[addr_reg]"r"(addr_reg)
          );
          break;
        }
        case 1: {
          __asm__ volatile(
          ".word 0x0055208b"
          : :[addr_reg]"r"(addr_reg)
          );
          break;
        }
        case 2: {
          __asm__ volatile(
          ".word 0x0055210b"
          : :[addr_reg]"r"(addr_reg)
          );
          break;
        }
        case 3: {
          __asm__ volatile(
          ".word 0x0055218b"
          : :[addr_reg]"r"(addr_reg)
          );
          break;
        }
        default: {  // case 4
          __asm__ volatile(
          ".word 0x0055220b"
          : :[addr_reg]"r"(addr_reg)
          );
          break;
        }
      }
      break;
    }
    default: {
      __asm__ volatile(
        ".word 0x0015200b"
        : :[addr_reg]"r"(addr_reg)
      );
      break;
    }
  }
}

inline void LoadV_D_Kernel(uint64_t addr, uint8_t k, uint8_t vec_addr, uint8_t vtag) {
  register uint64_t addr_reg asm ("a0") = addr;
  switch (k) {
    case 1: {
      __asm__ volatile(
        ".word 0x0015280b"
        : :[addr_reg]"r"(addr_reg)
      );
      break;
    }
    case 2: {
      switch (vec_addr) {
        case 0: {
          __asm__ volatile(
          ".word 0x0025280b"
          : :[addr_reg]"r"(addr_reg)
          );
          break;
        }
        default: {  // case 1
          __asm__ volatile(
          ".word 0x0025288b"
          : :[addr_reg]"r"(addr_reg)
          );
          break;
        }
      }
      break;
    }
    case 3: {
      switch (vec_addr) {
        case 0: {
          __asm__ volatile(
          ".word 0x0035280b"
          : :[addr_reg]"r"(addr_reg)
          );
          break;
        }
        case 1: {
          __asm__ volatile(
          ".word 0x0035288b"
          : :[addr_reg]"r"(addr_reg)
          );
          break;
        }
        default: {  // case 2
          __asm__ volatile(
          ".word 0x0035290b"
          : :[addr_reg]"r"(addr_reg)
          );
          break;
        }
      }
      break;
    }
    case 4: {
      switch (vec_addr) {
        case 0: {
          __asm__ volatile(
          ".word 0x0045280b"
          : :[addr_reg]"r"(addr_reg)
          );
          break;
        }
        case 1: {
          __asm__ volatile(
          ".word 0x0045288b"
          : :[addr_reg]"r"(addr_reg)
          );
          break;
        }
        case 2: {
          __asm__ volatile(
          ".word 0x0045290b"
          : :[addr_reg]"r"(addr_reg)
          );
          break;
        }
        default: {  // case 3
          __asm__ volatile(
          ".word 0x0045298b"
          : :[addr_reg]"r"(addr_reg)
          );
          break;
        }
      }
      break;
    }
    case 5: {
      switch (vec_addr) {
        case 0: {
          __asm__ volatile(
          ".word 0x0055280b"
          : :[addr_reg]"r"(addr_reg)
          );
          break;
        }
        case 1: {
          __asm__ volatile(
          ".word 0x0055288b"
          : :[addr_reg]"r"(addr_reg)
          );
          break;
        }
        case 2: {
          __asm__ volatile(
          ".word 0x0055290b"
          : :[addr_reg]"r"(addr_reg)
          );
          break;
        }
        case 3: {
          __asm__ volatile(
          ".word 0x0055298b"
          : :[addr_reg]"r"(addr_reg)
          );
          break;
        }
        default: {  // case 4
          __asm__ volatile(
          ".word 0x00552a0b"
          : :[addr_reg]"r"(addr_reg)
          );
          break;
        }
      }
      break;
    }
    default: {
      __asm__ volatile(
        ".word 0x0015280b"
        : :[addr_reg]"r"(addr_reg)
      );
      break;
    }
  }
}

inline void LoadV_P(uint64_t addr, uint8_t k, uint8_t vtag) {
  register uint64_t addr_reg asm ("a0") = addr;
  switch (k) {
    case 1: {
      __asm__ volatile(
        ".word 0x0015300b"
        : :[addr_reg]"r"(addr_reg)
      );
      break;
    }
    case 2: {
      __asm__ volatile(
        ".word 0x0025300b"
        : :[addr_reg]"r"(addr_reg)
      );
      break;
    }
    case 3: {
      __asm__ volatile(
        ".word 0x0035300b"
        : :[addr_reg]"r"(addr_reg)
      );
      break;
    }
    case 4: {
      __asm__ volatile(
        ".word 0x0045300b"
        : :[addr_reg]"r"(addr_reg)
      );
      break;
    }
    case 5: {
      __asm__ volatile(
        ".word 0x0055300b"
        : :[addr_reg]"r"(addr_reg)
      );
      break;
    }
    default: {
      __asm__ volatile(
        ".word 0x0015300b"
        : :[addr_reg]"r"(addr_reg)
      );
      break;
    }
  }
}

inline int32_t Conv(uint8_t k) {
  register int32_t res asm ("a0");
  switch (k) {
    case 1: {
      __asm__ volatile(
        ".word 0x0010450b"
        :[res]"=r"(res)
      );
      break;
    }
    case 2: {
      __asm__ volatile(
        ".word 0x0020450b"
        :[res]"=r"(res)
      );
      break;
    }
    case 3: {
      __asm__ volatile(
        ".word 0x0030450b"
        :[res]"=r"(res)
      );
      break;
    }
    case 4: {
      __asm__ volatile(
        ".word 0x0040450b"
        :[res]"=r"(res)
      );
      break;
    }
    case 5: {
      __asm__ volatile(
        ".word 0x0050450b"
        :[res]"=r"(res)
      );
      break;
    }
    default: {
      __asm__ volatile(
        ".word 0x0010450b"
        :[res]"=r"(res)
      );
      break;
    }
  }
  return res;
}

inline uint16_t Pool_Max(uint8_t k) {
  register uint16_t res asm ("a0");
  switch (k) {
    case 1: {
      __asm__ volatile(
        ".word 0x0010e50b"
        :[res]"=r"(res)
      );
      break;
    }
    case 2: {
      __asm__ volatile(
        ".word 0x0020e50b"
        :[res]"=r"(res)
      );
      break;
    }
    case 3: {
      __asm__ volatile(
        ".word 0x0030e50b"
        :[res]"=r"(res)
      );
      break;
    }
    case 4: {
      __asm__ volatile(
        ".word 0x0040e50b"
        :[res]"=r"(res)
      );
      break;
    }
    case 5: {
      __asm__ volatile(
        ".word 0x0050e50b"
        :[res]"=r"(res)
      );
      break;
    }
    default: {
      __asm__ volatile(
        ".word 0x0010e50b"
        :[res]"=r"(res)
      );
      break;
    }
  }
  return res;
}

inline uint16_t Pool_Avg(uint8_t k) {
  register uint16_t res asm ("a0");
  switch (k) {
    case 1: {
      __asm__ volatile(
        ".word 0x0011650b"
        :[res]"=r"(res)
      );
      break;
    }
    case 2: {
      __asm__ volatile(
        ".word 0x0021650b"
        :[res]"=r"(res)
      );
      break;
    }
    case 3: {
      __asm__ volatile(
        ".word 0x0031650b"
        :[res]"=r"(res)
      );
      break;
    }
    case 4: {
      __asm__ volatile(
        ".word 0x0041650b"
        :[res]"=r"(res)
      );
      break;
    }
    case 5: {
      __asm__ volatile(
        ".word 0x0051650b"
        :[res]"=r"(res)
      );
      break;
    }
    default: {
      __asm__ volatile(
        ".word 0x0011650b"
        :[res]"=r"(res)
      );
      break;
    }
  }
  return res;
}

inline uint64_t Act(uint64_t data, uint64_t zero) {
  register uint64_t data_reg asm ("a0") = data;
  register uint64_t zero_reg asm ("a1") = zero;
  register uint64_t res asm ("a2");
  __asm__ volatile(
    ".word 0x00b5760b"
    :[res]"=r"(res)
    :[data_reg]"r"(data_reg), [zero_reg]"r"(zero_reg)
  );
  return res;
}

#ifdef __cplusplus
}
#endif
