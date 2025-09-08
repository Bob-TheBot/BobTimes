import { cn } from "@/lib/utils";

interface SelaLogoProps {
  className?: string;
  width?: number;
  height?: number;
}

export function SelaLogo({ className, width = 91, height = 33 }: SelaLogoProps) {
  const dotSize = (6.3 / 91) * width;
  const dotTop = (29.5 / 33) * height - (3.14 / 33) * height;
  const dotRight = (3.2 / 91) * width;

  return (
    <div className={cn("flex items-center relative", className)}>
      <div className="relative">
        <img
          src="/sela_logo.svg"
          alt="Sela Logo"
          width={width}
          height={height}
          className="transition-all duration-200 dark:[filter:brightness(0)_invert(1)_contrast(200%)]"
        />
      </div>
    </div>
  );
}