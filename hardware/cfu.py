from amaranth import Array, Elaboratable, Module, ResetSignal, Signal
from amaranth.build import Platform


class Instruction(Elaboratable):
    """
    Custom instruction.
    This encodes a 32-bit R-type instruction with:
    - `opcode = 0x0B` (`CUSTOM_0`) — one of four opcode slots reserved for extensions
    - `funct3 = 0`, `funct7 = 0` — function selectors
    - `rd = a0`, `rs1 = a1`, `rs2 = a2` — register operands

    ```
      R-type instruction encoding (32 bits):

      31        25 24    20 19    15 14  12 11     7 6       0
      ┌──────────┬────────┬────────┬──────┬────────┬─────────┐
      │  funct7  │  rs2   │  rs1   │funct3│   rd   │ opcode  │
      │  7 bits  │ 5 bits │ 5 bits │3 bits│ 5 bits │ 7 bits  │
      └──────────┴────────┴────────┴──────┴────────┴─────────┘
      │ 0000000  │  a2    │  a1    │ 000  │  a0    │ 0001011 │
      └──────────┴────────┴────────┴──────┴────────┴─────────┘
                                                     CUSTOM_0
    """

    def __init__(self) -> None:
        self.in0 = Signal(32)
        self.in1 = Signal(32)
        self.funct7 = Signal(7)
        self.funct3 = Signal(3)
        self.output = Signal(32)
        self.start = Signal()
        self.done = Signal()

    def signal_done(self, m):
        m.d.comb += self.done.eq(1)

    def elaborate(self, platform: Platform):
        raise NotImplementedError()


class _FallbackInstruction(Instruction):
    """Executed by CFU when no instruction explicitly defined for a given function id.

    This does nothing useful, but it does ensure that the CFU does not hang on an unknown functionid.
    """

    def elaborate(self, platform: Platform):
        m = Module()
        m.d.comb += self.output.eq(self.in0)
        m.d.comb += self.done.eq(1)
        return m


class Cfu(Elaboratable):
    """
    Custom Function Unit

    RR instruction format, funct3 is used to distinguish 8 instructions.
    Names are fixed by the Litex Cfu bus
    """

    def __init__(self) -> None:
        self.cmd_valid = Signal(name="cmd_valid")
        self.cmd_ready = Signal(name="cmd_ready")
        self.cmd_function_id = Signal(
            10, name="cmd_payload_function_id"
        )  # funct3 + funct7
        self.cmd_in0 = Signal(32, name="cmd_payload_inputs_0")
        self.cmd_in1 = Signal(32, name="cmd_payload_inputs_1")
        self.rsp_valid = Signal(name="rsp_valid")
        self.rsp_ready = Signal(name="rsp_ready")
        self.rsp_out = Signal(32, name="rsp_payload_outputs_0")
        self.reset = Signal(name="reset")

        self.lram_addr = [Signal(32, name=f"port{i}_addr") for i in range(4)]
        self.lram_data = [Signal(32, name=f"port{i}_din") for i in range(4)]

        self.ports = (
            [
                self.cmd_valid,
                self.cmd_ready,
                self.cmd_function_id,
                self.cmd_in0,
                self.cmd_in1,
                self.rsp_valid,
                self.rsp_ready,
                self.rsp_out,
                self.reset,
            ]
            + self.lram_addr
            + self.lram_data
        )

    def elab_instructions(self, m):
        """
        define instructions + add them to submodules
        returns id : Module - where id is in range(8)
        """
        return dict()

    def __build_instructions(self, m):
        """Builds the list of eight instructions"""
        instruction_dict = self.elab_instructions(m)

        assert all(k in range(8) for k in instruction_dict.keys()), (
            "Instruction IDs must be integers from 0 to 7"
        )

        # Add fallback instructions where needed
        for i in range(8):
            if i not in instruction_dict:
                m.submodules[f"fallback{i}"] = fb = _FallbackInstruction()
                instruction_dict[i] = fb

        return list(instruction_dict[i] for i in range(8))

    def elaborate(self, platform: Platform):
        """
        FSM design
        ┌──────────┐   cmd_valid & done     ┌──────────────┐
        │          │   & rsp_ready          │              │
        │ WAIT_CMD │◄──────────────────────│ WAIT_TRANSFER│
        │          │                        │ (CPU not     │
        │ cmd_ready│   cmd_valid & done    │  ready yet)  │
        │ = 1      │   & !rsp_ready        │ rsp_valid=1  │
        │          │───────────────────────►│              │
        │          │                        └──────┬───────┘
        │          │   cmd_valid & !done           │ rsp_ready
        │          │──────────┐                    │
        └──────────┘          │            ┌───────┘
                               ▼            │
                      ┌──────────────┐      │
                      │              │      │
                      │WAIT_INSTRUCT.│──────┘
                      │ (multi-cycle │  done & !rsp_ready
                      │  instruction)│
                      │ cmd_ready=0  │──────┐
                      │              │      │ done & rsp_ready
                      └──────────────┘      │
                               ▲            │
                               │            ▼
                               │     ┌──────────┐
                               └─────│ WAIT_CMD │
                                     └──────────┘

        """
        m = Module()
        # Internal Wiring
        funct3 = Signal(3)
        funct7 = Signal(7)
        m.d.comb += [
            funct3.eq(self.cmd_function_id[:3]),
            funct7.eq(self.cmd_function_id[-7:]),
        ]
        stored_function_id = Signal(3)
        current_function_id = Signal(3)
        current_function_done = Signal()
        stored_output = Signal(32)

        instructions = self.__build_instructions(m)
        instruction_outputs = Array(Signal(32) for _ in range(8))
        instruction_dones = Array(Signal() for _ in range(8))
        instruction_starts = Array(Signal() for _ in range(8))
        for i, instruction in enumerate(instructions):
            m.d.comb += instruction_outputs[i].eq(instruction.output)
            m.d.comb += instruction_dones[i].eq(instruction.done)
            m.d.comb += instruction.start.eq(instruction_starts[i])

        def check_instruction_done():
            with m.If(current_function_done):
                m.d.comb += self.rsp_valid.eq(1)
                m.d.comb += self.rsp_out.eq(instruction_outputs[current_function_id])
                with m.If(self.rsp_ready):
                    m.next = "WAIT_CMD"
                with m.Else():
                    m.d.sync += stored_output.eq(
                        instruction_outputs[current_function_id]
                    )
                    m.next = "WAIT_TRANSFER"
            with m.Else():
                m.next = "WAIT_INSTRUCTION"

        with m.FSM():
            with m.State("WAIT_CMD"):
                # We're waiting for a command from the CPU.
                m.d.comb += current_function_id.eq(funct3)
                m.d.comb += current_function_done.eq(
                    instruction_dones[current_function_id]
                )
                m.d.comb += self.cmd_ready.eq(1)
                with m.If(self.cmd_valid):
                    m.d.sync += stored_function_id.eq(self.cmd_function_id[:3])
                    m.d.comb += instruction_starts[current_function_id].eq(1)
                    # Fast path: check if instruction completes this cycle (single-cycle instructions).
                    # If done, send result immediately or buffer it. If not done, move to WAIT_INSTRUCTION.
                    check_instruction_done()
            with m.State("WAIT_INSTRUCTION"):
                # An instruction is executing on the CFU. We're waiting until it
                # completes.
                m.d.comb += current_function_id.eq(stored_function_id)
                m.d.comb += current_function_done.eq(
                    instruction_dones[current_function_id]
                )
                check_instruction_done()
            with m.State("WAIT_TRANSFER"):
                # Instruction has completed, but the CPU isn't ready to receive
                # the result.
                m.d.comb += self.rsp_valid.eq(1)
                m.d.comb += self.rsp_out.eq(stored_output)
                with m.If(self.rsp_ready):
                    m.next = "WAIT_CMD"

        for instruction in instructions:
            m.d.comb += [
                instruction.in0.eq(self.cmd_in0),
                instruction.in1.eq(self.cmd_in1),
                instruction.funct7.eq(funct7),
            ]

        return m
