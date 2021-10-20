<?php

/**
 * This file is part of the Zephir.
 *
 * (c) Phalcon Team <team@zephir-lang.com>
 *
 * For the full copyright and license information, please view
 * the LICENSE file that was distributed with this source code.
 */

declare(strict_types=1);

namespace Zephir;

/**
 * CodePrinter.
 *
 * Buffers code, making it look pretty
 */
class CodePrinter
{
    protected string $code = '';

    protected string $lastLine = '';

    protected int $level = 0;

    protected int $currentPrints = 0;

    /**
     * Adds a line to the output without the automatic line feed.
     *
     * @param string $code
     */
    public function outputNoLineFeed(string $code): void
    {
        $this->lastLine = $code;
        $this->code .= str_repeat("\t", $this->level).$code;
    }

    /**
     * Add code to the output at the beginning.
     *
     * @param string $code
     */
    public function preOutput(string $code): void
    {
        $this->lastLine = $code;
        $this->code = str_repeat("\t", $this->level).$code.PHP_EOL.$this->code;
        ++$this->currentPrints;
    }

    /**
     * Adds a line to the output without the automatic line feed.
     *
     * @param string $code
     */
    public function preOutputNoLineFeed(string $code): void
    {
        $this->lastLine = $code;
        $this->code = str_repeat("\t", $this->level).$code.$this->code;
    }

    /**
     * Adds code to the output without the indentation level.
     *
     * @param string $code
     */
    public function preOutputNoLevel(string $code): void
    {
        $this->lastLine = $code;
        $this->code = $code.PHP_EOL.$this->code;
        ++$this->currentPrints;
    }

    /**
     * Add code to the output without indentation.
     *
     * @param string $code
     */
    public function outputNoIndent(string $code): void
    {
        $this->lastLine = $code;
        $this->code .= $code.PHP_EOL;
        ++$this->currentPrints;
    }

    /**
     * Add code to the output.
     *
     * @param string $code
     * @param bool $appendEOL
     */
    public function output(string $code, bool $appendEOL = true): void
    {
        $this->lastLine = $code;
        $this->code .= str_repeat("\t", $this->level).$code.($appendEOL ? PHP_EOL : '');
        ++$this->currentPrints;
    }

    /**
     * Adds a comment to the output with indentation level.
     *
     * @param $docblock
     * @param bool $replaceTab
     */
    public function outputDocBlock($docblock, bool $replaceTab = true): void
    {
        $code = '';
        $docblock = '/'.$docblock.'/';

        foreach (explode("\n", $docblock) as $line) {
            if ($replaceTab) {
                $code .= str_repeat("\t", $this->level).preg_replace('/^[ \t]+/', ' ', $line).PHP_EOL;
            } else {
                $code .= $line.PHP_EOL;
            }
        }

        $this->lastLine = $code;
        $this->code .= $code;
        ++$this->currentPrints;
    }

    /**
     * Adds code to the output without the indentation level.
     *
     * @param string $code
     */
    public function outputNoLevel(string $code): void
    {
        $this->lastLine = $code;
        $this->code .= $code.PHP_EOL;
        ++$this->currentPrints;
    }

    /**
     * Adds a blank line to the output
     * Optionally controlling if the blank link must be added if the
     * previous line added isn't one blank line too.
     *
     * @param bool $ifPrevNotBlank
     */
    public function preOutputBlankLine(bool $ifPrevNotBlank = false): void
    {
        if (!$ifPrevNotBlank) {
            $this->code = PHP_EOL.$this->code;
            $this->lastLine = PHP_EOL;
            ++$this->currentPrints;
        } else {
            if (trim($this->lastLine)) {
                $this->code = PHP_EOL.$this->code;
                $this->lastLine = PHP_EOL;
                ++$this->currentPrints;
            }
        }
    }

    /**
     * Adds a blank line to the output
     * Optionally controlling if the blank link must be added if the
     * previous line added isn't one blank line too.
     *
     * @param bool $ifPrevNotBlank
     */
    public function outputBlankLine(bool $ifPrevNotBlank = false): void
    {
        if (!$ifPrevNotBlank) {
            $this->code .= PHP_EOL;
            $this->lastLine = PHP_EOL;
            ++$this->currentPrints;
        } else {
            if (trim($this->lastLine)) {
                $this->code .= PHP_EOL;
                $this->lastLine = PHP_EOL;
                ++$this->currentPrints;
            }
        }
    }

    /**
     * Increase the indentation level.
     */
    public function increaseLevel(): void
    {
        ++$this->level;
    }

    /**
     * Decrease the indentation level.
     */
    public function decreaseLevel(): void
    {
        --$this->level;
    }

    public function setLevel(int $level): void
    {
        $this->level = $level;
    }

    /**
     * Returns the output in the buffer.
     *
     * @return string
     */
    public function getOutput(): string
    {
        return $this->code;
    }

    /**
     * Returns an approximate number of lines printed by the CodePrinter.
     *
     * @return int
     */
    public function getNumberPrints(): int
    {
        return $this->currentPrints;
    }

    /**
     * Frees memory used within the code.
     */
    public function clear(): void
    {
        $this->code = '';
        $this->lastLine = '';
        $this->level = 0;
    }

    public function duplicate(): CodePrinter
    {
        $printer = new self();
        $printer->setLevel($this->level);

        return $printer;
    }
}
