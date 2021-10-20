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

use Psr\Log\LoggerInterface;
use Zephir\Cache\FunctionCache;
use Zephir\Exception\CompilerException;
use Zephir\Passes\StaticTypeInference;

/**
 * CompilationContext.
 *
 * This class encapsulates important entities required during compilation
 */
class CompilationContext
{
    /**
     * @var EventsManager|null
     */
    public ?EventsManager $eventsManager = null;

    /**
     * Compiler.
     *
     * @var Compiler|null
     */
    public ?Compiler $compiler = null;

    /**
     * Current code printer.
     *
     * @var CodePrinter|null
     */
    public ?CodePrinter $codePrinter = null;

    /**
     * Whether the current method is static or not.
     *
     * @var bool
     */
    public bool $staticContext = false;

    /**
     * Code printer for the header.
     *
     * @var CodePrinter|null
     */
    public ?CodePrinter $headerPrinter = null;

    /**
     * Current symbol table.
     *
     * @var SymbolTable|null
     */
    public ?SymbolTable $symbolTable = null;

    /**
     * Type inference data.
     *
     * @var StaticTypeInference|null
     */
    public ?StaticTypeInference $typeInference = null;

    /**
     * Represents the class currently being compiled.
     *
     * @var ClassDefinition|null
     */
    public ?ClassDefinition $classDefinition = null;

    /**
     * Current method or function that being compiled.
     *
     * @var ClassMethod|FunctionDefinition|null
     */
    public ?ClassMethod $currentMethod = null;

    /**
     * Methods warm-up.
     *
     * @var MethodCallWarmUp|null
     */
    public ?MethodCallWarmUp $methodWarmUp = null;

    /**
     * Represents the c-headers added to the file.
     *
     * @var HeadersManager|null
     */
    public ?HeadersManager $headersManager = null;

    /**
     * Represents interned strings and concatenations made in the project.
     *
     * @var StringsManager|null
     */
    public ?StringsManager $stringsManager = null;

    /**
     * Tells if the the compilation is being made inside a cycle/loop.
     *
     * @var int
     */
    public int $insideCycle = 0;

    /**
     * Tells if the the compilation is being made inside a try/catch block.
     *
     * @var int
     */
    public int $insideTryCatch = 0;

    /**
     * Tells if the the compilation is being made inside a switch.
     *
     * @var int
     */
    public int $insideSwitch = 0;

    /**
     * Current cycle/loop block.
     *
     * @var array
     */
    public array $cycleBlocks = [];

    /**
     * The current branch, variables declared in conditional branches
     * must be market if they're used out of those branches.
     */
    public int $currentBranch = 0;

    /**
     * Global consecutive for try/catch blocks.
     *
     * @var int
     */
    public int $currentTryCatch = 0;

    /**
     * Helps to create graphs of conditional/jump branches in a specific method.
     *
     * @var BranchManager|null
     */
    public ?BranchManager $branchManager = null;

    /**
     * Manages both function and method call caches.
     *
     * @var CacheManager|null
     */
    public ?CacheManager $cacheManager = null;

    /**
     * Manages class renamings using keyword 'use'.
     *
     * @var AliasManager|null
     */
    public ?AliasManager $aliasManager = null;

    /**
     * Function Cache.
     *
     * @var FunctionCache|null
     */
    public ?FunctionCache $functionCache = null;

    /**
     * Global config.
     *
     * @var Config|null
     */
    public ?Config $config = null;

    /**
     * Global logger.
     *
     * @var LoggerInterface|null
     */
    public ?LoggerInterface $logger = null;

    /**
     * The current backend.
     *
     * @var BaseBackend|null
     */
    public ?BaseBackend $backend = null;

    /**
     * Transform class/interface name to FQN format.
     *
     * @param string $className
     *
     * @return string
     */
    public function getFullName(string $className): string
    {
        $isFunction = $this->currentMethod && $this->currentMethod instanceof FunctionDefinition;
        $namespace = $isFunction ? $this->currentMethod->getNamespace() : $this->classDefinition->getNamespace();

        return fqcn($className, $namespace, $this->aliasManager);
    }

    /**
     * Lookup a class from a given class name.
     *
     * @param string     $className
     * @param array|null $statement
     *
     * @return ClassDefinition
     */
    public function classLookup(string $className, array $statement = null): ClassDefinition
    {
        if (!\in_array($className, ['self', 'static', 'parent'])) {
            $className = $this->getFullName($className);
            if ($this->compiler->isClass($className)) {
                return $this->compiler->getClassDefinition($className);
            }

            throw new CompilerException("Cannot locate class '{$className}'", $statement);
        }

        if (\in_array($className, ['self', 'static'])) {
            return $this->classDefinition;
        }

        $parent = $this->classDefinition->getExtendsClass();
        if (!$parent instanceof ClassDefinition) {
            throw new CompilerException(
                sprintf(
                    'Cannot access parent:: because class %s does not extend any class',
                    $this->classDefinition->getCompleteName()
                ),
                $statement
            );
        }

        return $this->classDefinition->getExtendsClassDefinition();
    }
}
